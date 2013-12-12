#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <stdio.h>

#include <boost/timer.hpp>

#include <vector>

#include "Cloudject.hpp"
#include "CloudjectModel.hpp"

using namespace boost::filesystem;


class DescriptorTester
{
	typedef Cloudject<pcl::PointXYZRGB, pcl::FPFHSignature33> Cloudject;
	typedef CloudjectModel<pcl::PointXYZRGB, pcl::FPFHSignature33> CloudjectModel;

public:
	DescriptorTester(int numObjects, int numInstancesTrain, int numInstancesTest, float maxCorrespondenceThres)
	{
		// We suppose test and training the same no of classes
		m_NumObjects = numObjects;

		// We assume the same number (in each set) of instances per class
		m_NumInstancesTrain = numInstancesTrain;
		m_NumInstancesTest = numInstancesTest;
		
		m_MaxCorrespondenceThres = maxCorrespondenceThres;
	}


	void loadObjectViews(const char* path,
		std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& views)
	{
		if( exists( path ) )
		{
			boost::filesystem::
			directory_iterator end;
			directory_iterator iter(path);

			pcl::PCDReader reader;

			for( ; iter != end ; ++iter )
			{
				if ( !is_directory( *iter ) && iter->path().extension().string().compare(".pcd") == 0)
				{
					std::stringstream ss;

					pcl::PointCloud<pcl::PointXYZRGB>::Ptr object (new pcl::PointCloud<pcl::PointXYZRGB>);
					reader.read( iter->path().string(), *object );

					views.push_back(object);
				}
			}
		}
	
		//pcl::PCDReader reader;
		//for (int i = 0; i < numObjects * numInstances; i++)
		//{
		//	std::stringstream ss;
		//	ss << dataPath << i << ".pcd";

		//	pcl::PointCloud<pcl::PointXYZRGB>::Ptr object (new pcl::PointCloud<pcl::PointXYZRGB>);
		//	reader.read(ss.str().c_str(), *object);

		//	views.push_back(object);
		//}
	}


	float computeAccuracy(std::vector<int> categories)
	{
		int hits = 0;
		for (int i = 0; i < categories.size(); i++)
		{
			if ( categories[i] == ((int) i/m_NumInstancesTest) )
				hits++;
		}

		return ((float) hits) / categories.size();
	}


	void createCloudjectModels(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> views, int numObjects, int numInstances, 
		float leafSize, float pointRejectionThresh, float ratioRejectionThresh, float sigmaPenaltyThresh, std::vector<CloudjectModel>& models)
	{
		for (int i = 0; i < numObjects; i++)
		{
			CloudjectModel model(i, m_NumInstancesTrain, leafSize, pointRejectionThresh, ratioRejectionThresh, sigmaPenaltyThresh);
			for (int j = 0; j < numInstances; j++)
			{
				int idx = i*numInstances+j;
				model.addView(views[idx]);
			}

			models.push_back(model);
		}
	}


	void createCloudjects(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> views, float leafSize, std::vector<Cloudject>& cjs)
	{
		for (int i = 0; i < views.size(); i++)
		{
			Cloudject c(views[i], leafSize);
			//// DEBUG
			//if (/*(i / 4*6*2*3 == 2 ) &&*/ (i % (4*6) >= 20 && i % (4*6) <= 23))
			//	std::cout << c.getNumOfPointsInOriginalViewA() << " (" << c.getNumOfPointsInViewA() << ")" << std::endl;
			cjs.push_back(c);
		}
	}


	void describeCloudjectModels(std::vector<CloudjectModel>& models, float normalRadius, float fpfhRadius)
	{
		for (int i = 0; i < models.size(); i++)
		{
			models[i].describe(normalRadius, fpfhRadius);
		}
	}


	void describeCloudjects(std::vector<Cloudject>& cloudjects, float normalRadius, float fpfhRadius)
	{
		for (int i = 0; i < cloudjects.size(); i++)
		{
			cloudjects[i].describe(normalRadius, fpfhRadius);
		}
	}


	void categorize(std::vector<CloudjectModel> models, std::vector<Cloudject> cloudjects, std::vector<int>& categories)
	{
		categories.resize(cloudjects.size(), -1); // initialize with "errors" (-1 values)

		for (int i = 0; i < cloudjects.size(); i++)
		{
			float maxScore = 0;
			//float minDist = std::numeric_limits<float>::infinity();
			//std::cout << i << ": ";
			for (int m = 0; m < models.size(); m++)
			{
				float score = models[m].match(cloudjects[i]);
				//std::cout << score << " ";
				if (score > maxScore)
				{
					categories[i] = m;
					maxScore = score;
				}
			}
			//std::cout << std::endl;
			//system("pause");

			// TODO: Rejection implementation
			// if (minDist > m_MaxCorrespondenceThres)
			// ... assign an arbitrary class ID
		}

		//for (int i = 0; i < categories.size(); i++)
		//{
		//	std::cout << categories[i] << " ";
		//}
		//std::cout << std::endl;
		//system("pause");
	}


void run()
	{
		std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> trViews, teViews;
		
		std::cout << "Loading training ... " << std::endl;
		loadObjectViews("../data/TrainPCDs/", trViews);

		std::cout << "Loading testing ... " << std::endl;
		loadObjectViews("../data/TestPCDs/", teViews);

		std::cout << "Categorizing ... " << std::endl;

		// Summary 1
		//static const float modelLeafSizes[] = {0.030, 0.015};
		//static const float leafSizes[] = {0.030, 0.0225, 0.015};
		//static const float normalsRadius[] = {0.06, 0.045, 0.030, 0.015};
		//static const float fpfhRadius[] = {0.300, 0.200, 0.150, 0.125, 0.100, 0.075, 0.050};

		// Best
		static const float modelLeafSizes[] = {0.015};
		static const float leafSizes[] = {0.0225};
		static const float normalsRadius[] = {0.06};
		static const float fpfhRadius[] = {0.125};

		static const float pointRejectionThresh[] = {1.0};
		static const float ratioRejectionThresh[] = {0.99, 0.995, 0.999};//0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 1};
		static const float sigmaPenaltyThresh[] = { /*0.095, 0.096, 0.097, 0.098, 0.099, */0.1/*, 0.11, 0.12, 0.13, 0.14, 0.15*/};
		
		//static const float modelLeafSizes[] = {0.030}; //, 0.010};
		//static const float leafSizes[] = {0.030};//{0.010, 0.015, 0.020, 0.03};
		//static const float normalsRadius[] = {0.08, 0.07, 0.06}; //, 0.04, 0.03, 0.02, 0.01};//{0.01, 0.03, 0.06};
		//static const float fpfhRadius[] = {0.30, 0.20}; //{0.100, 0.075, 0.050, 0.025};// {0.025, 0.05, 0.075, 0.1};

		std::vector<float> vModelLeafSizes (modelLeafSizes, modelLeafSizes + sizeof(modelLeafSizes) / sizeof(modelLeafSizes[0]) );
		std::vector<float> vLeafSizes (leafSizes, leafSizes + sizeof(leafSizes) / sizeof(leafSizes[0]) );
		std::vector<float> vNormalsRadius (normalsRadius, normalsRadius + sizeof(normalsRadius) / sizeof(normalsRadius[0]) );
		std::vector<float> vFpfhRadius (fpfhRadius, fpfhRadius + sizeof(fpfhRadius) / sizeof(fpfhRadius[0]) );

		std::vector<float> vPointRejectionThresh (pointRejectionThresh, pointRejectionThresh + sizeof(pointRejectionThresh) / sizeof(pointRejectionThresh[0]) );
		std::vector<float> vRatioRejectionThresh (ratioRejectionThresh, ratioRejectionThresh + sizeof(ratioRejectionThresh) / sizeof(ratioRejectionThresh[0]) );
		std::vector<float> vSigmaPenaltyThresh (sigmaPenaltyThresh, sigmaPenaltyThresh + sizeof(sigmaPenaltyThresh) / sizeof(sigmaPenaltyThresh[0]) );

		// results summary (accuracy and description and categorization times)
		std::ofstream pSummaryFile ("../results/summary.txt", std::ios::out | std::ios::app);
		std::ofstream pPredsFile ("../results/predictions.txt", std::ios::out | std::ios::app);

		pSummaryFile << "sigma ratio point leaf_model leaf normal fpfh acc descr_time categ_time" << std::endl;

		for (int s = 0; s < vSigmaPenaltyThresh.size(); s++)
		for (int r = 0; r < vRatioRejectionThresh.size(); r++) 
		for (int p = 0; p < vPointRejectionThresh.size(); p++)
		{
		for (int m = 0; m < vModelLeafSizes.size(); m++)
		{
			for (int i = 0; i < vLeafSizes.size(); i++)
			{
				for (int j = 0; j < vNormalsRadius.size(); j++)
				{
					for (int k = 0; k < vFpfhRadius.size(); k++)
					{		
						std::cout << vSigmaPenaltyThresh[s] << " "
								  << vRatioRejectionThresh[r] << " " << vPointRejectionThresh[p] << " "
								  << vModelLeafSizes[m] << " " << vLeafSizes[i] << " " 
							      << vNormalsRadius[j] << " " << vFpfhRadius[k] << " ";

						// Downsample train
						std::vector<CloudjectModel> cjModels;
						createCloudjectModels(trViews, m_NumObjects, m_NumInstancesTrain, vModelLeafSizes[m], 
							vPointRejectionThresh[p], vRatioRejectionThresh[r], vSigmaPenaltyThresh[s], cjModels);
						// Downsample test
						std::vector<Cloudject> cjs;
						createCloudjects(teViews, vLeafSizes[i], cjs); 

						float accuracy;
						double descriptionTime, categorizationTime;

						std::vector<int> predictions;

						// No sense to have worse models than tests, or to have bigger radius (normals or fpfh) than the voxels' leaf size.
						if (   vModelLeafSizes[m] > vLeafSizes[i] 
							|| vNormalsRadius[j] < vLeafSizes[i]      || vFpfhRadius[k] < vLeafSizes[i]
							|| vNormalsRadius[j] < vModelLeafSizes[m] || vFpfhRadius[k] < vModelLeafSizes[m] )
						{
							accuracy = 0;
							descriptionTime = 0;
							categorizationTime = 0;
							predictions.resize(cjs.size(), -1);
						}
						else
						{
							boost::timer t;

							t.restart();
							describeCloudjectModels(cjModels, vNormalsRadius[j], vFpfhRadius[k]);
							describeCloudjects(cjs, vNormalsRadius[j], vFpfhRadius[k]);
							descriptionTime = t.elapsed();
					
							t.restart();
							categorize(cjModels, cjs, predictions);
							categorizationTime = t.elapsed();

							accuracy = computeAccuracy(predictions);
						}

						std::cout << accuracy << " " << descriptionTime << " " << categorizationTime << std::endl;

						// Save to file

						std::stringstream summaryline;
						summaryline << vSigmaPenaltyThresh[s] << " "
								    << vRatioRejectionThresh[r] << " " << vPointRejectionThresh[p] << " "
								    << vModelLeafSizes[m] << " " << vLeafSizes[i] << " " 
							        << vNormalsRadius[j] << " " << vFpfhRadius[k] << " " 
								    << accuracy << " " << descriptionTime << " " << categorizationTime << std::endl;
						pSummaryFile << summaryline.str();

						std::copy(predictions.begin(), predictions.end(), std::ostream_iterator<int>(pPredsFile, " "));
						pPredsFile << std::endl;
					}
				}
			}
		}
		}
	}

private:
	int m_NumObjects;

	int m_NumInstancesTrain;
	int m_NumInstancesTest;

	float m_MaxCorrespondenceThres;

};





int main(int argc, const char argv[])
{
	// Num objects, num view per model, num of instances of test per object, and max distance for views correspondence
	// numInstancesofTestPerObject: 4*6 number of positions in the grid, 3 number of views per object, and 2 instances of each view in each position
	DescriptorTester dt(5, 3, 4*6*3*2, 60);
	dt.run();
	system("pause");
	return 0;
}