#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/console/parse.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/timer.hpp>
#include <boost/assign/std/vector.hpp>

#include <vector>
#include <string>

#include "Cloudject.hpp"
#include "CloudjectModel.hpp"

#include <iterator>
#include <iostream>
#include <algorithm>

#include "xtlcommon.h"
#include "xtio.h"
#include "xtmath.h"
#include "xtvalidation.h"


using namespace boost::filesystem;
using namespace boost::assign;
using namespace std;

#ifdef __APPLE__
#define PARENT_DIR                "../../"
#elif _WIN32 || _WIN64
#define PARENT_DIR                "../"
#endif

#define NUM_OF_PARTITIONS         399
#define NUM_OF_PARTITIONS_VAL     5
#define SEED                      71
#define SUMMARY_FILEPATH          "summary.txt"
#define SCORES_FILEPATH           "scores.txt"


enum {FPFH, PFHRGB};


////////////////////////////////////////////////////////////////////////////////

class Line {
    std::string data;
public:
    friend std::istream &operator>>(std::istream &is, Line &l) {
        std::getline(is, l.data);
        return is;
    }
    operator std::string() const { return data; }
};

////////////////////////////////////////////////////////////////////////////////

template<typename SignatureT>
class DescriptorTester
{

	typedef LFCloudject<pcl::PointXYZRGB, SignatureT> Cloudject;
	typedef typename LFCloudject<pcl::PointXYZRGB, SignatureT>::Ptr CloudjectPtr;
	typedef LFCloudjectModel<pcl::PointXYZRGB, SignatureT> CloudjectModel;
	typedef typename LFCloudjectModel<pcl::PointXYZRGB, SignatureT>::Ptr CloudjectModelPtr;
    
public:
    
    DescriptorTester(vector<string> modelsNames, vector<int> modelsNumOfViews, int gheight, int gwidth)
    {
        m_ModelsNames = modelsNames;
        m_ModelsNumOfViews = modelsNumOfViews;
        m_GridHeight = gheight;
        m_GridWidth = gwidth;
    }
    
    void setModelsDataPath(const char* path)
    {
        m_ModelsPath = std::string(path);
    }
    
    void setTestDataPath(const char* path)
    {
        m_TestPath = std::string(path);
    }
    
    void setOutputFilePaths(const char* ofSummariesPath, const char* ofScoresPath)
    {
        m_OfSummariesPath = std::string(ofSummariesPath);
        m_OfScoresPath = std::string(ofScoresPath);
    }

	void computeScores(vector<vector<float> > params)
	{
		vector<vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > trViews, teViews;
		
        // Load the point clouds (cloudjects' views)
        
		cout << "Loading training ... " << endl;
		loadObjectViews(m_ModelsPath.c_str(), "PCDs/", m_ModelsNames, trViews);

		cout << "Loading testing ... " << endl;
		loadObjectViews(m_TestPath.c_str(), "PCDs/", m_ModelsNames, teViews);

        // Construct the cloudjects
        
        vector<CloudjectModelPtr> models;
        createCloudjectModels(trViews, m_ModelsNames, models);
        
        vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> teViewsSrl;
        vector<int> groundtruth;
        for (int i = 0; i < m_ModelsNames.size(); i++)
        {
            teViewsSrl.reserve(teViewsSrl.size() + teViews[i].size());
            teViewsSrl.insert(teViewsSrl.end(), teViews[i].begin(), teViews[i].end());
            
            vector<int> objectLabels (m_ModelsNumOfViews[i] * m_GridHeight * m_GridWidth, i);
            groundtruth.reserve(groundtruth.size() + objectLabels.size());
            groundtruth.insert(groundtruth.end(), objectLabels.begin(), objectLabels.end());
        }
        
        vector<CloudjectPtr> cloudjects;
        createCloudjects(teViewsSrl, cloudjects);
        
		cout << "Computing scores ... " << endl;
        
        boost::timer t;

		vector<float> vLeafSizes = params[0];
		vector<float> vNormalsRadius = params[1];
		vector<float> vPfhRadius = params[2];
		vector<float> vPointRejectionThresh = params[3];

		// results summary (accuracy and description and categorization times)
		ofstream summariesFile (m_OfSummariesPath.c_str(), ios::out);
        ofstream scoresFile (m_OfScoresPath.c_str(), ios::out);

		summariesFile << "leaf normal pfh point acc descr_time dist_time categ_time" << endl;
        
        for (int i = 0; i < vLeafSizes.size(); i++)
        {
            downsampleCloudjectModels(models, vLeafSizes[i]);
            downsampleCloudjects(cloudjects, vLeafSizes[i]);
            
            for (int j = 0; j < vNormalsRadius.size(); j++)
            for (int k = 0; k < vPfhRadius.size(); k++)
            {
                t.restart();
                // (t)
                describeCloudjectModels(models, vNormalsRadius[j], vPfhRadius[k]);
                describeCloudjects(cloudjects, vNormalsRadius[j], vPfhRadius[k]);
                // (t)
                float descriptionTime = t.elapsed();
                
                t.restart();
                // (t)
                vector<vector<vector<vector<float> > > > distances;
                if ( vNormalsRadius[j] > vLeafSizes[i] && vPfhRadius[k] > vLeafSizes[i] && vPfhRadius[k] > vNormalsRadius[j])
                {
                    getCloudjectsDistancesToModels(models, cloudjects, distances);
                }
                // (t)
                float distancesComputationTime = t.elapsed();
                
                for (int p = 0; p < vPointRejectionThresh.size(); p++)
                {		
                    cout    << vLeafSizes[i] << " "
                            << vNormalsRadius[j] << " "
                            << vPfhRadius[k] << " "
                            << vPointRejectionThresh[p] << " " << endl;

                    setCloudjectModelsParameters(models, 0, 0, 0, 0);

                    float categorizationTime;

                    vector<vector<float> > scores;
                    
                    // No sense to have worse models than tests, or to have bigger radius (normals or fpfh) than the voxels' leaf size.
                    if ( vNormalsRadius[j] > vLeafSizes[i] && vPfhRadius[k] > vLeafSizes[i] && vPfhRadius[k] > vNormalsRadius[j])
                    {
                        t.restart();
                        // (t)
                        getScores(cloudjects, distances, scores, vPointRejectionThresh[p]);
                        // (t)
                        categorizationTime = t.elapsed();
                    }
                    else
                    {
                        descriptionTime = 0;
                        distancesComputationTime = 0;
                        categorizationTime = 0;
                    }

                    cout    << descriptionTime << " "
                            << distancesComputationTime << " "
                            << categorizationTime << endl << endl;

                    // Save to file

                    std::vector<float> parameters;
                    parameters += vLeafSizes[i], vNormalsRadius[j], vPfhRadius[k], vPointRejectionThresh[p], descriptionTime, distancesComputationTime, categorizationTime;
                    copy(parameters.begin(), parameters.end(), ostream_iterator<float>(summariesFile, " "));
                    summariesFile << endl;
                    
                    for (int c = 0; c < scores.size(); c++)
                    {
                        copy(scores[c].begin(), scores[c].end(), ostream_iterator<float>(scoresFile, " "));
                        scoresFile << endl;
                    } scoresFile << endl;
                }
            }
        }
	}
    
    void validateWithoutRejection(std::vector<int> dontcareIndices)
    {
        std::vector<std::vector<float> > summaries;
        loadSummariesFromFile(m_OfSummariesPath, summaries, true);
        
        std::vector<std::vector<std::vector<float> > > scores;
        loadScoresFromFile(m_OfScoresPath, scores);
        
        std::vector<int> groundtruth;
        for (int m = 0; m < m_ModelsNumOfViews.size(); m++)
        {
            std::vector<int> objInstanceIdx (m_ModelsNumOfViews[m] * m_GridHeight * m_GridWidth, m);
            groundtruth.reserve(groundtruth.size() + objInstanceIdx.size());
            groundtruth.insert(groundtruth.end(), objInstanceIdx.begin(), objInstanceIdx.end());
        }
        
        std::vector<float> rjtThreshsBest (m_ModelsNames.size(), 0);
        std::vector<bool> dcs = xtl::indicesToLogicals(m_ModelsNames.size(), dontcareIndices);
        
        // Out-of-sample
        xtl::LoocvPartition cvpst (groundtruth.size(), SEED);
        
        std::vector<int> oosPredictions, oosGroundtruth;
        oosPredictions.reserve(cvpst.getNumOfPartitions());
        oosGroundtruth.reserve(cvpst.getNumOfPartitions());
        
        vector<vector<float> > paramsAccs (cvpst.getNumOfPartitions());
        
        for (int p = 0; p < cvpst.getNumOfPartitions(); p++)
        {
            cout << p << " ";

            std::vector<int> groundtruthTr, groundtruthTe;
            cvpst.getTrain(groundtruth, p, groundtruthTr);
            cvpst.getTest (groundtruth, p, groundtruthTe);

            std::vector<float> paramsInternalAccs (summaries.size());
            for (int i = 0; i < summaries.size(); i++)
            {
                if (scores[i].empty()) continue;

                // Train
                std::vector<std::vector<float> > scoresTr;
                cvpst.getTrain(scores[i], p, scoresTr);

                vector<int> indicesTr;
                xtl::filterByValue(groundtruthTr, dontcareIndices, groundtruthTr, indicesTr, false);
                xtl::filterByIndex(scoresTr, indicesTr, vector<int>(), scoresTr, true, false);
                
                std::vector<int> predictionsTr;
                classify(scoresTr, rjtThreshsBest, dcs, predictionsTr);
                
                paramsInternalAccs[i] = xtl::computeAccuracy(groundtruthTr, predictionsTr);
            }
            
            paramsAccs[p] = paramsInternalAccs;

            int idxValBest;
            float accValBest = 0.f;
            for (int i = 0; i < summaries.size(); i++)
            {
                float acc = paramsInternalAccs[i];
                if (acc > accValBest)
                {
                    idxValBest = i;
                    accValBest = acc;
                }
            }
            
            cout << accValBest << endl;

            std::vector<std::vector<float> > scoresTe;
            cvpst.getTest(scores[idxValBest], p, scoresTe);

            std::vector<int> indicesTe;
            xtl::filterByValue(groundtruthTe, dontcareIndices, groundtruthTe, indicesTe, false);
            xtl::filterByIndex(scoresTe, indicesTe, vector<int>(), scoresTe, true, false);
            
            std::vector<int> predictionsTe;
            classify(scoresTe, rjtThreshsBest, dcs, predictionsTe);
            
            if (predictionsTe.size() > 0)
            {
                oosPredictions.insert(oosPredictions.end(), predictionsTe.begin(), predictionsTe.end());
                oosGroundtruth.insert(oosGroundtruth.end(), groundtruthTe.begin(), groundtruthTe.end());
            }
        } std::cout << std::endl;
        
        // Print groundtruth
        std::cout << "Groundtruth (labels)" << endl;
        std::copy(oosGroundtruth.begin(), oosGroundtruth.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
        
        // Print out-of-sample predictions
        std::cout << "Predictions (labels)" << endl;
        std::copy(oosPredictions.begin(), oosPredictions.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
        
        // Print confusion matrix
        std::cout << "Confusion matrix" << endl;
        std::vector<std::vector<float> > cnfMat;
        xtl::computeConfusion(oosGroundtruth, oosPredictions, cnfMat, true); // normalized
        xtl::print(cnfMat);

        // Get the out of sample accuracy of the outer cv
        float accMean = xtl::computeAccuracy(oosGroundtruth, oosPredictions);
        cout << "Out-of-sample accuracy [0,1]: " << accMean << endl;
        cout << endl;
        
        vector<float> paramsAccsAvg;
        xtl::collapse(paramsAccs, 0, paramsAccsAvg, xtl::COLLAPSE_AVG);
        
        // Show combinations' performance print
        cout << "Combinations performance: " << endl;
        xtl::print(paramsAccsAvg);
        cout << endl;
        
        // Index of best combination performance in average
        std::pair<int,float> paramsMaxAcc = xtl::max(paramsAccsAvg);
        
        cout << "Best combination perfomance (#" << paramsMaxAcc.first << "): ";
        xtl::print(summaries[paramsMaxAcc.first]);
        cout << endl;
    }
    
    void validateWithRejection(std::vector<float> objRejThreshs, std::vector<int> dontcareIndices)
    {
        std::vector<std::vector<float> > summaries;
        loadSummariesFromFile(m_OfSummariesPath, summaries, true);
        
        std::vector<std::vector<std::vector<float> > > scores;
        loadScoresFromFile(m_OfScoresPath, scores);
        
        std::vector<int> groundtruth;
        for (int m = 0; m < m_ModelsNumOfViews.size(); m++)
        {
            std::vector<int> objInstanceIdx (m_ModelsNumOfViews[m] * m_GridHeight * m_GridWidth, m);
            groundtruth.reserve(groundtruth.size() + objInstanceIdx.size());
            groundtruth.insert(groundtruth.end(), objInstanceIdx.begin(), objInstanceIdx.end());
        }
        
        std::vector<float> rjtThreshsBest (m_ModelsNames.size(), 0);
        std::vector<bool> dcs = xtl::indicesToLogicals(m_ModelsNames.size(), dontcareIndices);
        
        xtl::LoocvPartition cvpst (groundtruth.size(), SEED);
        
        // Out-of-sample accuracy
        vector<int> oosPredictions, oosGroundtruth;
        oosPredictions.reserve(cvpst.getNumOfPartitions());
        oosGroundtruth.reserve(cvpst.getNumOfPartitions());
        
        vector<vector<float> > paramsAccs (cvpst.getNumOfPartitions());
        for (int p = 0; p < cvpst.getNumOfPartitions(); p++)
        {
            cout << p << " ";

            vector<int> groundtruthTr, groundtruthTe;
            cvpst.getTrainTest(groundtruth, p, groundtruthTr, groundtruthTe);
            
            xtl::CvPartition icvpst (groundtruthTr, NUM_OF_PARTITIONS_VAL, SEED);
            boost::timer t;
            vector<vector<float> > paramsInternalsAccs (icvpst.getNumOfPartitions());
            for (int pp = 0; pp < icvpst.getNumOfPartitions(); pp++)
            {
                //            cout << p << " : " << pp << endl;
                vector<int> groundtruthTrTr, groundtruthTrVal;
                icvpst.getTrainTest(groundtruthTr, pp, groundtruthTrTr, groundtruthTrVal);
                
                paramsInternalsAccs[pp].resize(summaries.size(), 0);
                for (int i = 0; i < summaries.size(); i++)
                {
                    if (scores[i].empty()) continue;
                    
                    // Train
                    vector<vector<float> > scoresTr;
                    cvpst.getTrain(scores[i], p, scoresTr);
                    
                    vector<vector<float> > scoresTrTr, scoresTrVal;
                    icvpst.getTrainTest(scoresTr, pp, scoresTrTr, scoresTrVal);
                    
                    vector<float> rjtThreshsBest, rjtAccuraciesBest;
                    trainRejectionThreshold(scoresTrTr, groundtruthTrTr, objRejThreshs, rjtThreshsBest, rjtAccuraciesBest);
                    
                    vector<int> predictionsTrVal;
                    vector<bool> _dcs (rjtThreshsBest.size());
                    classify(scoresTrVal, rjtThreshsBest, _dcs, predictionsTrVal);
                    paramsInternalsAccs[pp][i] = xtl::computeAccuracy(groundtruthTrVal, predictionsTrVal);
                }
            }
            
            xtl::collapse(paramsInternalsAccs, 0, paramsAccs[p], xtl::COLLAPSE_AVG);
            
            int idxValBest;
            float accValBest = 0.f;
            for (int i = 0; i < summaries.size(); i++)
            {
                float accAcc = 0.f;
                for (int pp = 0; pp < icvpst.getNumOfPartitions(); pp++)
                    accAcc += paramsInternalsAccs[pp][i];
                float acc = accAcc / icvpst.getNumOfPartitions();
                
                if (acc > accValBest)
                {
                    idxValBest = i;
                    accValBest = acc;
                }
            }
            
            cout << accValBest << endl;
            
            vector<vector<float> > scoresTr, scoresTe;
            cvpst.getTrainTest(scores[idxValBest], p, scoresTr, scoresTe);
            
            vector<float> rjtThreshsBest;
            vector<float> rjtAccuraciesBest;
            trainRejectionThreshold(scoresTr, groundtruthTr, objRejThreshs, rjtThreshsBest, rjtAccuraciesBest);
            
            // Replace dontcare class labels by -1 to match the rejectin prediciton (-1)
            vector<int> aux;
            vector<int> indices;
            xtl::filterByValue(groundtruthTe, dontcareIndices, aux, indices);
            aux = groundtruthTe;
            for (int i = 0; i < indices.size(); i++)
                aux[indices[i]] = -1;
            
            vector<int> predictionsTe;
            classify(scoresTe, rjtThreshsBest, dcs, predictionsTe);
            
            oosGroundtruth.insert(oosGroundtruth.end(), aux.begin(), aux.end());
            oosPredictions.insert(oosPredictions.end(), predictionsTe.begin(), predictionsTe.end());
        } std::cout << std::endl;
        
        // Print groundtruth
        std::cout << "Groundtruth (labels)" << endl;
        std::copy(oosGroundtruth.begin(), oosGroundtruth.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
        
        // Print out-of-sample predictions
        std::cout << "Predictions (labels)" << endl;
        std::copy(oosPredictions.begin(), oosPredictions.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
        
        // Print confusion matrix
        std::cout << "Confusion matrix" << endl;
        std::vector<std::vector<float> > cnfMat;
        xtl::computeConfusion(oosGroundtruth, oosPredictions, cnfMat, true); // normalized
        xtl::print(cnfMat);
        
        // Get the out of sample accuracy of the outer cv
        float accMean = xtl::computeAccuracy(oosGroundtruth, oosPredictions);
        cout << "Out-of-sample accuracy [0,1]: " << accMean << endl;
        cout << endl;
        
        vector<float> paramsAccsAvg;
        xtl::collapse(paramsAccs, 0, paramsAccsAvg, xtl::COLLAPSE_AVG);
        
        // Show combinations' performance print
        cout << "Combinations performance: " << endl;
        xtl::print(paramsAccsAvg);
        cout << endl;
        
        // Index of best combination performance in average
        std::pair<int,float> paramsMaxAcc = xtl::max(paramsAccsAvg);
        
        cout << "Best combination perfomance (#" << paramsMaxAcc.first << "): ";
        xtl::print(summaries[paramsMaxAcc.first]);
        
        cout << "Rejection thresholds combination in best performance (#" << paramsMaxAcc.first << "): ";
        std::vector<float> objRejThreshsBest, objRejAccsBest;
        trainRejectionThreshold(scores[paramsMaxAcc.first], groundtruth, objRejThreshs, objRejThreshsBest, objRejAccsBest);
        xtl::print(objRejThreshsBest);
        xtl::print(objRejAccsBest);
    }

private:
    
    void trainRejectionThreshold(vector<vector<float> > train, vector<int> labels, vector<float> objRejThreshs, vector<float>& threshs, vector<float>& traccuracies)
    {
        int numOfModels = train[0].size();
        
        std::vector<std::vector<int> > gg (numOfModels, std::vector<int>(train.size()));
        for (int k = 0; k < numOfModels; k++)
        {
            for (int i = 0; i < train.size(); i++)
                gg[k][i] = (labels[i] == k) ? 1 : (-1);
        }
        
        threshs.resize(numOfModels);
        traccuracies.resize(numOfModels);
        for (int k = 0; k < numOfModels; k++)
        {
            int rjtIdxBest;
            float rjtAccBest = 0.f;
            for (int t = 0; t < objRejThreshs.size(); t++)
            {
                std::vector<int> p (train.size());
                for (int i = 0; i < train.size(); i++)
                {
                    p[i] = (train[i][k] > objRejThreshs[t]) ? 1 : (-1);
                }
                
                float acc = xtl::computeAccuracy(gg[k],p);
                if (acc > rjtAccBest)
                {
                    rjtIdxBest = t;
                    rjtAccBest = acc;
                }
            }
            
            threshs[k] = objRejThreshs[rjtIdxBest];
            traccuracies[k] = rjtAccBest;
        }
    }
    
    void classify(std::vector<std::vector<float> > test, std::vector<float> rjtThreshs, std::vector<bool> dontcares, std::vector<int>& predictions)
    {
        predictions.resize(test.size(), -1);
        
        for (int i = 0; i < test.size(); i++)
        {
            int maxIdx = -1;
            float maxMargin = 0;
            
            for (int k = 0; k < test[i].size(); k++)
            {
                float score = test[i][k];
                float margin = score - rjtThreshs[k];
                if ( (score > rjtThreshs[k]) && (margin > maxMargin) && !dontcares[k] )
                {
                    maxIdx = k;
                    maxMargin = margin;
                }
            }
            
            predictions[i] = maxIdx;
        }
    }
    
    void createCloudjectModels(vector<vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > views, vector<string> names, vector<CloudjectModelPtr>& models)
	{
        assert (names.size() == views.size());
        
        models.resize(names.size());
		for (int i = 0; i < names.size(); i++)
            models[i] = CloudjectModelPtr(new CloudjectModel(i, names[i], views[i]));
	}
    
	void createCloudjects(vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> views, vector<CloudjectPtr>& cjs)
	{
        cjs.resize(views.size());
        for (int i = 0; i < views.size(); i++)
            cjs[i] = CloudjectPtr(new Cloudject(views[i]));
	}
    
    void downsampleCloudjectModels(vector<CloudjectModelPtr> models, float leafSize)
    {
        for (int i = 0; i < models.size(); i++)
			models[i]->downsample(leafSize);
    }
    
    void downsampleCloudjects(vector<CloudjectPtr> cloudjects, float leafSize)
    {
        for (int i = 0; i < cloudjects.size(); i++)
			cloudjects[i]->downsample(leafSize);
    }
	
	void setCloudjectModelsParameters(vector<CloudjectModelPtr> models, float pointRejectionThresh, float ratioRejectionThresh, int penaltyMode, float sigmaPenaltyThresh)
	{
		for (int m = 0; m < models.size(); m++)
		{
            models[m]->setPointScoreRejectionThreshold(pointRejectionThresh);
            models[m]->setPointRatioRejectionThreshold(ratioRejectionThresh);
            models[m]->setSizePenaltyMode(penaltyMode);
            models[m]->setSigmaPenaltyThreshold(sigmaPenaltyThresh);
		}
	}
    
	void describeCloudjectModels(vector<CloudjectModelPtr> models, float normalRadius, float PfhRadius)
	{
		for (int i = 0; i < models.size(); i++)
			models[i]->describe(normalRadius, PfhRadius);
	}
    
	void describeCloudjects(vector<CloudjectPtr> cloudjects, float normalRadius, float PfhRadius)
	{
		for (int i = 0; i < cloudjects.size(); i++)
			cloudjects[i]->describe(normalRadius, PfhRadius);
	}
    
    
	void getCloudjectsDistancesToModels(vector<CloudjectModelPtr> models, vector<CloudjectPtr> cloudjects, vector<vector<vector<vector<float> > > >& distances)
	{
		distances.resize(cloudjects.size());
		for (int i = 0; i < cloudjects.size(); i++)
		{
			distances[i].resize(models.size());
			for (int m = 0; m < models.size(); m++)
			{
                distances[i][m];
				models[m]->getMinimumDistances(cloudjects[i], distances[i][m]);
			}
		}
	}
    
    float distToScore(float dist)
    {
        return 1 - dist;
    }
    
	void getScores(vector<CloudjectPtr> cloudjects, vector<vector<vector<vector<float> > > >& distances, vector<vector<float> >& scores, float ptRejectionThresh = 0.f)
	{
        scores.resize(distances.size(), vector<float>(distances[0].size()));
		for (int i = 0; i < distances.size(); i++)
		{
			float maxScore = 0.f;
            
			for (int m = 0; m < distances[i].size(); m++)
			{
                float wtScoreAcc = 0.f;
                float invdistToCameraAcc = 0.f;
                
                // #views
                for (int v = 0; v < distances[i][m].size(); v++)
                {
                    float scoreViewAcc = 0.f;
                    int positives = 0;
                    
                    for (int p = 0; p < distances[i][m][v].size(); p++)
                    {
                        float scoreVal = distToScore(distances[i][m][v][p]);
                        if (scoreVal > ptRejectionThresh)
                        {
                            scoreViewAcc += scoreVal;
                            positives ++;
                        }
                    }
                    
                    float scoreAcc = (positives == 0) ? 0 : (scoreViewAcc / positives);
                    
                    pcl::PointXYZ pos = cloudjects[i]->getPosition(v);
                    float distToCamera = sqrt(pow(pos.x,2) + pow(pos.y,2) + pow(pos.z,2));
                    
                    // Weight the view score proportionally to the distance to camera
                    wtScoreAcc += (1.f/distToCamera * scoreAcc);
                    invdistToCameraAcc += (1.f/distToCamera);
                }
                
                float finalScore = wtScoreAcc / invdistToCameraAcc;
                scores[i][m] = finalScore;
			}
 		}
	}
    
    void loadSummariesFromFile(std::string filePath, std::vector<std::vector<float> >& matrix, bool header = false)
    {
        ifstream file (filePath.c_str());
        
        matrix.clear();
        if (file.is_open())
        {
            // Read the lines of the file
            std::vector<std::string> lines;
            std::copy(std::istream_iterator<Line>(file),
                      std::istream_iterator<Line>(),
                      std::back_inserter(lines));
            
            // First line is a header
            matrix.resize(header ? (lines.size() - 1) : lines.size());
            for (int i = 0; i < matrix.size(); i++)
            {
                vector<string> fields;
                boost::split(fields, lines[header ? (i+1) : i], boost::is_any_of(" "));
                
                matrix[i].resize(fields.size() - 1);
                for (int j = 0; j < fields.size() - 1; j++)
                    matrix[i][j] = stof(fields[j]);
            }
        }
    }
    
    void loadScoresFromFile(std::string filePath, std::vector<std::vector<std::vector<float> > >& matrices, bool header = false)
    {
        ifstream file (filePath.c_str());
        
        matrices.clear();
        if (file.is_open())
        {
            // Read the lines of the file
            std::vector<std::string> lines;
            std::copy(std::istream_iterator<Line>(file),
                      std::istream_iterator<Line>(),
                      std::back_inserter(lines));
            
            // First line is a header
            std::vector<std::vector<float> > matrix;
            for (int i = 0; i < (header ? (lines.size() - 1) : lines.size()); i++)
            {
                vector<string> fields;
                boost::split(fields, lines[header ? (i+1) : i], boost::is_any_of(" "));
                
                if (fields.size() - 1 == 0)
                {
                    matrices.push_back(matrix);
                    
                    matrix.clear();
                }
                else
                {
                    vector<float> row (fields.size() - 1);
                    for (int j = 0; j < fields.size() - 1; j++)
                        row[j] = stof(fields[j]);
                    
                    matrix.push_back(row);
                }
            }
        }
    }
    
    void saveSummariesToFile(string filePath, std::vector<std::vector<float> > summaries)
    {
        ofstream summaryFile (filePath.c_str(), ios::out);
        
        for (int i = 0; i < summaries.size(); i++)
        {
            copy(summaries[i].begin(), summaries[i].end(), ostream_iterator<float>(summaryFile, " "));
            summaryFile << endl;
        }
        
        summaryFile.close();
    }
    
    void saveScoresToFile(string filePath, std::vector<std::vector<std::vector<float> > > scores)
    {
        ofstream scoresFile (filePath.c_str(), ios::out);
        
        for (int i = 0; i < scores.size(); i++)
        {
            for (int m = 0; m < scores[i].size(); m++)
            {
                copy(scores[i][m].begin(), scores[i][m].end(), ostream_iterator<float>(scoresFile, " "));
                scoresFile << endl;
            }
            scoresFile << endl;
        }
        
        scoresFile.close();
    }
    
    void loadObjectViews(const char* path, const char* pcdDir, vector<string> modelsNames,
                         vector<vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >& views)
	{
        pcl::PCDReader reader; // to read pcd files containing point clouds
        
        views.resize(modelsNames.size());
        for (int i = 0; i < modelsNames.size(); i++)
        {
            string objectPath = path + modelsNames[i] + "/" + pcdDir;
            vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> objectViews;
            
            if( exists( objectPath ) )
            {
                boost::filesystem::directory_iterator end;
                boost::filesystem::directory_iterator iter(objectPath);
                
                for( ; iter != end ; ++iter )
                {
                    if ( !is_directory( *iter ) && iter->path().extension().string().compare(".pcd") == 0)
                    {
                        stringstream ss;
                        
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr object (new pcl::PointCloud<pcl::PointXYZRGB>);
                        reader.read( iter->path().string(), *object );
                        
                        objectViews.push_back(object);
                    }
                }
            }
            
            views[i] = objectViews;
        }
	}
    
    std::string m_ModelsPath;
    std::string m_TestPath;
    std::vector<string> m_ModelsNames;
    std::vector<int> m_ModelsNumOfViews;
    int m_GridHeight;
    int m_GridWidth;
    
    std::string m_OfSummariesPath, m_OfScoresPath;
};

////////////////////////////////////////////////////////////////////////////////


void splitString(std::string str, std::string separator, std::vector<int>& args)
{
    // Preprocessing the arguments
    std::vector<std::string> LStr;
    boost::split(LStr, str, boost::is_any_of(separator));
    
    for (vector<string>::iterator it = LStr.begin(); it != LStr.end(); ++it)
        args.push_back(stoi(*it));
}

void splitString(std::string str, std::string separator, std::vector<float>& args)
{
    // Preprocessing the arguments
    std::vector<std::string> LStr;
    boost::split(LStr, str, boost::is_any_of(separator));
    
    for (vector<string>::iterator it = LStr.begin(); it != LStr.end(); ++it)
        args.push_back(stof(*it));
}

void splitString(std::string str, std::string separator, std::vector<string>& args)
{
    args.clear();
    boost::split(args, str, boost::is_any_of(separator));
}

int main(int argc, char** argv)
{
    //-------------------------------------------------------------------//
    // *---------------------------------------------------------------*
    // *    FIND THE BEST COMBINATION OF PARAMETERS (EXCEPT REJECTION)
    // *---------------------------------------------------------------*
    //-------------------------------------------------------------------//
    
    //
    // The system need to be parametrized
    //
    
    // ./descriptors_test -m "book,cup,dish,grimper,scissors,pillbox,tetrabrick" -v "3,1,3,2,2,5,3" -g 7,3 -d 0
    //
    //      -d: option is 0 for FPFH and 1 for PFHRGB
    
    //
    // Optional arguments came later:
    //
    
    // Specify the output paths
    //   -O "summary.txt,scores.txt"
    // Specify the arguments for the computation of scores
    //   -S -t "../../data/Models/" -e "../../data/Test/" -d 0 -l "0.01,0.02,0.03" -n "0.03,0.05,0.07" -f "0.05,0.075,0.10,0.125,0.15,0.20" -p 0,1,0.1
    // Choose the validation
    // -V (-i "3,4")
    // -Vr "0,0.025,1" -i "3,4", validation with rejection
    
    
    int pos;
    
    // Parse system-related arguments

    std::string modelsNamesStr;
    pcl::console::parse_argument(argc, argv, "-m", modelsNamesStr);
    std::vector<std::string> modelsNamesLStr;
    splitString(modelsNamesStr, ",", modelsNamesLStr);
    
    std::string modelsNumOfViewsStr;
    pcl::console::parse_argument(argc, argv, "-v", modelsNumOfViewsStr);
    std::vector<int> modelsNumOfViews;
    splitString(modelsNumOfViewsStr, ",", modelsNumOfViews);
    
    int gheight, gwidth;
    pcl::console::parse_2x_arguments(argc, argv, "-g", gheight, gwidth);
    
    int descriptionType;
    pcl::console::parse(argc, argv, "-d", descriptionType);
    
    void* pDt;
    ////////////////////////////////////////////////////////////////////////
    // Initialization
    if (descriptionType == FPFH)
        pDt = (void*) new DescriptorTester<pcl::FPFHSignature33>(modelsNamesLStr, modelsNumOfViews, gheight, gwidth);
    else if (descriptionType == PFHRGB)
        pDt = (void*) new DescriptorTester<pcl::PFHRGBSignature250>(modelsNamesLStr, modelsNumOfViews, gheight, gwidth);
    ////////////////////////////////////////////////////////////////////////
    
    // Parse 'output file'-related arguments
    
    bool bFilenamesParsed = false;
    string ofSummariesPath, ofScoresPath;
    if ( (pos = pcl::console::find_argument(argc, argv, "-O")) >= 0 )
    {
        std::string ofPathsStr;
        pcl::console::parse_argument(argc, argv, "-O", ofPathsStr);
        std::vector<std::string> ofPaths;
        splitString(ofPathsStr, ",", ofPaths);
        
        assert(ofPaths.size() == 2);
        
        ofSummariesPath = ofPaths[0];
        ofScoresPath = ofPaths[1];
        
        bFilenamesParsed = true;
    }
    
    ////////////////////////////////////////////////////////////////////////
    // Set parameters to the system
    if (bFilenamesParsed)
    {
        if (descriptionType == FPFH)
            ((DescriptorTester<pcl::FPFHSignature33>*) pDt)->setOutputFilePaths(ofSummariesPath.c_str(), ofScoresPath.c_str());
        else if (descriptionType == PFHRGB)
            ((DescriptorTester<pcl::PFHRGBSignature250>*) pDt)->setOutputFilePaths(ofSummariesPath.c_str(), ofScoresPath.c_str());
    }
    else
    {
        if (descriptionType == FPFH)
            ((DescriptorTester<pcl::FPFHSignature33>*) pDt)->setOutputFilePaths(SUMMARY_FILEPATH, SCORES_FILEPATH);
        else if (descriptionType == PFHRGB)
            ((DescriptorTester<pcl::PFHRGBSignature250>*) pDt)->setOutputFilePaths(SUMMARY_FILEPATH, SCORES_FILEPATH);
    }
    ////////////////////////////////////////////////////////////////////////

    // Parse data and score computation-related parameters
    
    if ( (pos = pcl::console::find_argument(argc, argv, "-S")) >= 0 )
    {
        // Parsing (not descriptor dependent)
        
        std::string modelsPathStr;
        pcl::console::parse_argument(argc, argv, "-t", modelsPathStr);
        
        std::string testPathStr;
        pcl::console::parse_argument(argc, argv, "-e", testPathStr);
        
        // Parsing (descriptor dependent)
        
        std::vector<std::vector<float> > params;
        
        if (descriptionType == FPFH || descriptionType == PFHRGB)
        {
            std::string leafSizesStr;
            pcl::console::parse_argument(argc, argv, "-l", leafSizesStr);
            
            std::string normalRadiusStr;
            pcl::console::parse_argument(argc, argv, "-n", normalRadiusStr);
            
            std::string pfhRadiusStr;
            pcl::console::parse_argument(argc, argv, "-f", pfhRadiusStr);
            std::vector<std::string> pfhRadiusLStr;
            boost::split(pfhRadiusLStr, pfhRadiusStr, boost::is_any_of(","));
            
            float ptRjThreshStart, ptRjThreshStep, ptRjThreshEnd;
            pcl::console::parse_3x_arguments(argc, argv, "-p", ptRjThreshStart, ptRjThreshEnd, ptRjThreshStep);
            
            std::vector<float> leafSizes;
            splitString(leafSizesStr, ",", leafSizes);
            
            std::vector<float> normalRadius;
            splitString(normalRadiusStr, ",", normalRadius);
            
            std::vector<float> pfhRadius;
            splitString(pfhRadiusStr, ",", pfhRadius);
            
            std::vector<float> ptRjtThreshs = xtl::linspace(ptRjThreshStart, ptRjThreshEnd, ptRjThreshStep, true);
            
            params += leafSizes, normalRadius, pfhRadius, ptRjtThreshs;
        }
        
        ////////////////////////////////////////////////////////////////////////
        // More settings
        if (descriptionType == FPFH)
        {
            ((DescriptorTester<pcl::FPFHSignature33>*) pDt)->setModelsDataPath(modelsPathStr.c_str());
            ((DescriptorTester<pcl::FPFHSignature33>*) pDt)->setTestDataPath(testPathStr.c_str());
        }
        else if (descriptionType == PFHRGB)
        {
            ((DescriptorTester<pcl::PFHRGBSignature250>*) pDt)->setModelsDataPath(modelsPathStr.c_str());
            ((DescriptorTester<pcl::PFHRGBSignature250>*) pDt)->setTestDataPath(testPathStr.c_str());
        }
        ////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // Computations of scores
        if (descriptionType == FPFH)
            ((DescriptorTester<pcl::FPFHSignature33>*) pDt)->computeScores(params);
        else if (descriptionType == PFHRGB)
            ((DescriptorTester<pcl::PFHRGBSignature250>*) pDt)->computeScores(params);
        ////////////////////////////////////////////////////////////////////////

    }

    // Validate without rejection
    
    if ( (pos = pcl::console::find_argument(argc, argv, "-V")) >= 0 )
    {
        // Parse dontcares (optional)
        vector<int> dontcareIndices;
        if ( (pos = pcl::console::find_argument(argc, argv, "-i")) >= 0 )
        {
            std::string dontcareIndicesStr;
            pcl::console::parse_argument(argc, argv, "-i", dontcareIndicesStr);
            
            splitString(dontcareIndicesStr, ",", dontcareIndices);
        }
        
        // Run the validation
        
        if (descriptionType == FPFH)
            ((DescriptorTester<pcl::FPFHSignature33>*) pDt)->validateWithoutRejection(dontcareIndices);
        else if (descriptionType == PFHRGB)
            ((DescriptorTester<pcl::PFHRGBSignature250>*) pDt)->validateWithoutRejection(dontcareIndices);
    }
    
    // Validate with rejection
    
    if ( (pos = pcl::console::find_argument(argc, argv, "-Vr")) >= 0 )
    {
        // Parse rejection thresholds
        float objRjThreshStart, objRjThreshStep, objRjThreshEnd;
        pcl::console::parse_3x_arguments(argc, argv, "-Vr", objRjThreshStart, objRjThreshStep, objRjThreshEnd);
        
        vector<float> objRejThreshs = xtl::linspace(objRjThreshStart, objRjThreshEnd, objRjThreshStep, false);
        
        // Parse dontcares (optional)
        vector<int> dontcareIndices;
        if ( (pos = pcl::console::find_argument(argc, argv, "-i")) >= 0 )
        {
            std::string dontcareIndicesStr;
            pcl::console::parse_argument(argc, argv, "-i", dontcareIndicesStr);
            
            splitString(dontcareIndicesStr, ",", dontcareIndices);
        }
        
        // Run the validation
        if (descriptionType == FPFH)
            ((DescriptorTester<pcl::FPFHSignature33>*) pDt)->validateWithRejection(objRejThreshs, dontcareIndices);
        else if (descriptionType == PFHRGB)
            ((DescriptorTester<pcl::PFHRGBSignature250>*) pDt)->validateWithRejection(objRejThreshs, dontcareIndices);
    }
    
    if (descriptionType == FPFH)
        delete ((DescriptorTester<pcl::FPFHSignature33>*) pDt);
    else if (descriptionType == PFHRGB)
        delete ((DescriptorTester<pcl::PFHRGBSignature250>*) pDt);
        
	return 0;
}