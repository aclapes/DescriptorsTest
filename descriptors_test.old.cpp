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


	void run()
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

        static const float leafSizes[] = {0.01, 0.02, 0.03};
        static const float normalsRadius[] = {0.03, 0.05, 0.075};
        static const float pfhRadius[] = {0.05, 0.075, 0.1, 0.15, 0.2};

		static const float pointRejectionThresh[] = {1.0};//{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
		static const float ratioRejectionThresh[] = {0.7, 0.8, 0.9, 1};
		static const int sizePenaltyMode[] = {0, 1, 2};
		static const float sigmaPenaltyThresh[] = { 5, 10, 15, 20 };

		vector<float> vLeafSizes (leafSizes, leafSizes + sizeof(leafSizes) / sizeof(leafSizes[0]) );
		vector<float> vNormalsRadius (normalsRadius, normalsRadius + sizeof(normalsRadius) / sizeof(normalsRadius[0]) );
		vector<float> vPfhRadius (pfhRadius, pfhRadius + sizeof(pfhRadius) / sizeof(pfhRadius[0]) );
		vector<float> vPointRejectionThresh (pointRejectionThresh, pointRejectionThresh + sizeof(pointRejectionThresh) / sizeof(pointRejectionThresh[0]) );
		vector<float> vRatioRejectionThresh (ratioRejectionThresh, ratioRejectionThresh + sizeof(ratioRejectionThresh) / sizeof(ratioRejectionThresh[0]) );
		vector<int> vSizePenaltyMode (sizePenaltyMode, sizePenaltyMode + sizeof(sizePenaltyMode) / sizeof(sizePenaltyMode[0]) );
		vector<float> vSigmaPenaltyThresh (sigmaPenaltyThresh, sigmaPenaltyThresh + sizeof(sigmaPenaltyThresh) / sizeof(sigmaPenaltyThresh[0]) );

		// results summary (accuracy and description and categorization times)
        string summaryFilePath = string(PARENT_DIR) + "results/summary.txt";
        string predsFilePath = string(PARENT_DIR) + "results/predictions.txt";
        string scoresFilePath = string(PARENT_DIR) + "results/scores.txt";
		ofstream pSummaryFile (summaryFilePath.c_str(), ios::out);
		ofstream pPredsFile (predsFilePath.c_str(), ios::out);
        ofstream pScoresFile (scoresFilePath.c_str(), ios::out);

		pSummaryFile << "leaf normal pfh point ratio penalty sigma acc descr_time categ_time" << endl;
        
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
                for (int r = 0; r < vRatioRejectionThresh.size(); r++)
                for (int z = 0; z < vSizePenaltyMode.size(); z++)
                for (int s = 0; s < vSigmaPenaltyThresh.size(); s++)
                {		
                    cout    << vLeafSizes[i] << " "
                            << vNormalsRadius[j] << " "
                            << vPfhRadius[k] << " "
                            << vPointRejectionThresh[p] << " "
                            << vRatioRejectionThresh[r] << " "
                            << vSizePenaltyMode[z] << " "
                            << vSigmaPenaltyThresh[s] << " ";

                    setCloudjectModelsParameters(models, vPointRejectionThresh[p], vRatioRejectionThresh[r], vSizePenaltyMode[z], vSigmaPenaltyThresh[s]);

                    float accuracy;
                    float categorizationTime;

                    vector<int> predictions;
                    vector<vector<float> > scores;

                    // No sense to have worse models than tests, or to have bigger radius (normals or fpfh) than the voxels' leaf size.
                    if ( vNormalsRadius[j] <= vLeafSizes[i] || vPfhRadius[k] <= vLeafSizes[i]
                        || vPfhRadius[k] <= vNormalsRadius[j] || (vSizePenaltyMode[z] == 0 && s > 0)  )
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
                    parameters += vLeafSizes[i], vNormalsRadius[j], vPfhRadius[k], vPointRejectionThresh[p], vRatioRejectionThresh[r], vSizePenaltyMode[z], vSigmaPenaltyThresh[s], accuracy, descriptionTime, categorizationTime;
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

	dt.run();

	return 0;
}