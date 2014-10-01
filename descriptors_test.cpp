#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <stdio.h>

#include <boost/timer.hpp>
#include <boost/assign/std/vector.hpp>

#include <vector>

#include "Cloudject.hpp"
#include "CloudjectModel.hpp"

using namespace boost::filesystem;
using namespace boost::assign;
using namespace std;

#ifdef __APPLE__
#define PARENT_DIR                "../../"
#elif _WIN32 || _WIN64
#define PARENT_DIR                "../"
#endif

#define FPFH_DESCRIPTION

class DescriptorTester
{
#ifdef FPFH_DESCRIPTION
	typedef LFCloudject<pcl::PointXYZRGB, pcl::FPFHSignature33> Cloudject;
	typedef LFCloudject<pcl::PointXYZRGB, pcl::FPFHSignature33>::Ptr CloudjectPtr;
	typedef LFCloudjectModel<pcl::PointXYZRGB, pcl::FPFHSignature33> CloudjectModel;
	typedef LFCloudjectModel<pcl::PointXYZRGB, pcl::FPFHSignature33>::Ptr CloudjectModelPtr;
#else
	typedef LFCloudject<pcl::PointXYZRGB, pcl::PFHRGBSignature250> Cloudject;
	typedef LFCloudject<pcl::PointXYZRGB, pcl::PFHRGBSignature250>::Ptr CloudjectPtr;
	typedef LFCloudjectModel<pcl::PointXYZRGB, pcl::PFHRGBSignature250> CloudjectModel;
	typedef LFCloudjectModel<pcl::PointXYZRGB, pcl::PFHRGBSignature250>::Ptr CloudjectModelPtr;
#endif

public:
    
    DescriptorTester(string modelsPath, string testPath, vector<string> modelsNames, vector<int> modelsNumOfViews, int gheight, int gwidth)
    {
        m_ModelsPath = modelsPath;
        m_TestPath = testPath;
        m_ModelsNames = modelsNames;
        m_ModelsNumOfViews = modelsNumOfViews;
        m_GridHeight = gheight;
        m_GridWidth = gwidth;
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

	float computeAccuracy(vector<int> groundtruth, vector<int> predictions)
	{
        assert (predictions.size() == groundtruth.size());
        
        vector<int> nhits;
        vector<int> ninstances;
        
        for (int i = 0; i < groundtruth.size(); ++i)
        {
            while (groundtruth[i] >= ninstances.size())
            {
                nhits.push_back(0);
                ninstances.push_back(0);
            }
            
            if (predictions[i] == groundtruth[i]) nhits[groundtruth[i]]++;
            ninstances[groundtruth[i]]++;
        }
    
        float wtHitsAcc = 0.f; // weighted hits accumulated
        for (int i = 0; i < ninstances.size(); i++)
            wtHitsAcc += ( ((float) nhits[i]) / ninstances[i] );

        return wtHitsAcc / ninstances.size();
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

	void categorize(vector<CloudjectModelPtr> models, vector<CloudjectPtr> cloudjects, vector<int>& categories, vector<vector<float> >& scores)
	{
		categories.resize(cloudjects.size(), -1); // initialize with "errors" (-1 values)
        scores.resize(cloudjects.size(), vector<float>(models.size()));
		for (int i = 0; i < cloudjects.size(); i++)
		{
			float maxScore = 0;

			for (int m = 0; m < models.size(); m++)
			{
				float score = models[m]->getScore(cloudjects[i]);

				if (score > maxScore)
				{
					categories[i] = m;
					maxScore = score;
				}
                
                scores[i][m] = score;
			}
		}
	}


	void run(vector<vector<float> > params)
	{
		vector<vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > trViews, teViews;
		
        // Load the point clouds (cloudjects' views)
        
		cout << "Loading training ... " << endl;
        string trainPath = string(PARENT_DIR) + "data/Models/";
		loadObjectViews(trainPath.c_str(), "PCDs/", m_ModelsNames, trViews);

		cout << "Loading testing ... " << endl;
        string testPath = string(PARENT_DIR) + "data/Test/";
		loadObjectViews(testPath.c_str(), "PCDs/", m_ModelsNames, teViews);

        // Construct the cloudjects
        
        vector<CloudjectModelPtr> models;
        createCloudjectModels(trViews, m_ModelsNames, models);
        
        vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> teViewsSrl;
        vector<int> groundtruth;
        for (int i = 0; i < m_ModelsNames.size(); i++)
        {
            teViewsSrl.reserve(teViewsSrl.size() + teViews[i].size());
            teViewsSrl.insert(teViewsSrl.begin(), teViews[i].begin(), teViews[i].end());
            
            vector<int> objectLabels (m_ModelsNumOfViews[i] * m_GridHeight * m_GridWidth, i);
            groundtruth.reserve(groundtruth.size() + objectLabels.size());
            groundtruth.insert(groundtruth.begin(), objectLabels.begin(), objectLabels.end());
        }
        
        vector<CloudjectPtr> cloudjects;
        createCloudjects(teViewsSrl, cloudjects);
        
		cout << "Categorizing ... " << endl;

		vector<float> vLeafSizes = params[0];
		vector<float> vNormalsRadius = params[1];
		vector<float> vPfhRadius = params[2];
		vector<float> vPointRejectionThresh = params[3];

		// results summary (accuracy and description and categorization times)
        string summaryFilePath = "summary.txt";
        string predsFilePath = "predictions.txt";
        string scoresFilePath = "scores.txt";
		ofstream pSummaryFile (summaryFilePath.c_str(), ios::out);
		ofstream pPredsFile (predsFilePath.c_str(), ios::out);
        ofstream pScoresFile (scoresFilePath.c_str(), ios::out);

		pSummaryFile << "leaf normal pfh point acc descr_time categ_time" << endl;
        
        for (int i = 0; i < vLeafSizes.size(); i++)
        {
            downsampleCloudjectModels(models, vLeafSizes[i]);
            downsampleCloudjects(cloudjects, vLeafSizes[i]);
            
            for (int j = 0; j < vNormalsRadius.size(); j++)
            for (int k = 0; k < vPfhRadius.size(); k++)
            {
                boost::timer t;
                t.restart();
                // (t)
                describeCloudjectModels(models, vNormalsRadius[j], vPfhRadius[k]);
                describeCloudjects(cloudjects, vNormalsRadius[j], vPfhRadius[k]);
                // (t)
                float descriptionTime = t.elapsed();
                
                for (int p = 0; p < vPointRejectionThresh.size(); p++)
                {		
                    cout    << vLeafSizes[i] << " "
                            << vNormalsRadius[j] << " "
                            << vPfhRadius[k] << " "
                            << vPointRejectionThresh[p] << " ";

                    setCloudjectModelsParameters(models, vPointRejectionThresh[p], 0, 0, 0);

                    float accuracy;
                    float categorizationTime;

                    vector<int> predictions;
                    vector<vector<float> > scores;

                    // No sense to have worse models than tests, or to have bigger radius (normals or fpfh) than the voxels' leaf size.
                    if ( vNormalsRadius[j] <= vLeafSizes[i] || vPfhRadius[k] <= vLeafSizes[i]
                        || vPfhRadius[k] <= vNormalsRadius[j])
                    {
                        accuracy = 0;
                        descriptionTime = 0;
                        categorizationTime = 0;
                        predictions.resize(cloudjects.size(), -1);
                    }
                    else
                    {
                        t.restart();
                        // (t)
                        categorize(models, cloudjects, predictions, scores);
                        // (t)
                        categorizationTime = t.elapsed();

                        accuracy = computeAccuracy(groundtruth, predictions);
                    }

                    cout    << accuracy << " "
                            << descriptionTime << " "
                            << categorizationTime << endl;

                    // Save to file

                    vector<float> parameters;
                    parameters += vLeafSizes[i], vNormalsRadius[j], vPfhRadius[k], vPointRejectionThresh[p], accuracy, descriptionTime, categorizationTime;
                    copy(parameters.begin(), parameters.end(), ostream_iterator<float>(pSummaryFile, " "));
                    pSummaryFile << endl;

                    copy(predictions.begin(), predictions.end(), ostream_iterator<int>(pPredsFile, " "));
                    pPredsFile << endl;
                    
                    for (int c = 0; c < scores.size(); c++)
                    {
                        copy(scores[c].begin(), scores[c].end(), ostream_iterator<float>(pScoresFile, " "));
                        pScoresFile << endl;
                    } pScoresFile << endl;
                }
            }
        }
	}

private:
//	int m_NumObjects;
//
//	int m_NumInstancesTrain;
//	int m_NumInstancesTest;
    
    string m_ModelsPath;
    string m_TestPath;
    vector<string> m_ModelsNames;
    vector<int> m_ModelsNumOfViews;
    int m_GridHeight;
    int m_GridWidth;
};

int main(int argc, char** argv)
{
	// Num objects, num view per model, num of instances of test per object
	// numInstancesofTestPerObject: 4*6 number of positions in the grid, 3 number of views per object, and 2 instances of each view in each position
//	DescriptorTester dt(5, 3, 4*6*3*2);
    
    string modelsPath = string(argv[1]);
    string testPath = string(argv[2]);
    
    vector<string> modelsNamesLStr, modelsNumOfViewsLStr;
    boost::split(modelsNamesLStr, argv[3], boost::is_any_of(","));
    boost::split(modelsNumOfViewsLStr, argv[4], boost::is_any_of(","));
    
    vector<int> modelsNumOfViews;
    for (vector<string>::iterator it = modelsNumOfViewsLStr.begin(); it != modelsNumOfViewsLStr.end(); ++it)
    {
        modelsNumOfViews.push_back(stoi(*it));
    }
    
    int gheight = atoi(argv[5]);
    int gwidth = atoi(argv[6]);
    
    DescriptorTester dt (modelsPath, testPath, modelsNamesLStr, modelsNumOfViews, gheight, gwidth);
    
    vector<string> leafSizesLStr, normalRadiusLStr, pfhRadiusLStr, ptRejThreshsLStr;
    boost::split(leafSizesLStr, argv[7], boost::is_any_of(","));
    boost::split(normalRadiusLStr, argv[8], boost::is_any_of(","));
    boost::split(pfhRadiusLStr, argv[9], boost::is_any_of(","));
    boost::split(ptRejThreshsLStr, argv[10], boost::is_any_of(","));
    
    vector<vector<float> > validationParams;
    
    vector<float> leafSizes, normalRadius, pfhRadius, ptRejThreshs;
    for (vector<string>::iterator it = leafSizesLStr.begin(); it != leafSizesLStr.end(); ++it)
        leafSizes.push_back(stof(*it));
    for (vector<string>::iterator it = normalRadiusLStr.begin(); it != normalRadiusLStr.end(); ++it)
        normalRadius.push_back(stof(*it));
    for (vector<string>::iterator it = pfhRadiusLStr.begin(); it != pfhRadiusLStr.end(); ++it)
        pfhRadius.push_back(stof(*it));
    for (vector<string>::iterator it = ptRejThreshsLStr.begin(); it != ptRejThreshsLStr.end(); ++it)
        ptRejThreshs.push_back(stof(*it));
    
    validationParams.push_back(leafSizes);
    validationParams.push_back(normalRadius);
    validationParams.push_back(pfhRadius);
    validationParams.push_back(ptRejThreshs);

	dt.run(validationParams);

	return 0;
}