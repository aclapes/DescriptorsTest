#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <stdio.h>

#include <boost/timer.hpp>

#include <vector>

#include "Cloudject.hpp"
#include "CloudjectModel.hpp"

using namespace boost::filesystem;

class FPFHCategorizer
{
	typedef pcl::PointXYZRGB PointT;
	typedef pcl::Normal	NormalT;
	typedef pcl::FPFHSignature33 SignatureT;
	typedef pcl::PointCloud<PointT> PointCloud;
	typedef pcl::PointCloud<NormalT> NormalCloud;
	typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
	typedef pcl::PointCloud<NormalT>::Ptr NormalCloudPtr;
	typedef pcl::PointCloud<SignatureT> Descriptor;
	typedef pcl::PointCloud<SignatureT>::Ptr DescriptorPtr;
	typedef CloudjectModel<PointT, SignatureT> CloudjectModel;
	typedef Cloudject<PointT, SignatureT> Cloudject;

public:
	FPFHCategorizer(float normalRadius, float fpfhRadius)
		: m_NormalRadius(normalRadius), m_FpfhRadius(fpfhRadius)
	{

	}

	void setTrainCloudjects(std::vector<CloudjectModel> models)
	{
		m_TrModels = models;
	}

	void extractNormalsFromView(PointCloudPtr pCloud, 
		NormalCloudPtr pNormals)
	{
		// Create the normal estimation class, and pass the input dataset to it
		pcl::NormalEstimation<PointT, NormalT> ne;
		ne.setInputCloud (pCloud);

		// Create an empty kdtree representation, and pass it to the normal estimation object.
		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
		pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
		ne.setSearchMethod (tree);

		// Output datasets
		// *initialized outside

		// Use all neighbors in a sphere of radius 3cm
		ne.setRadiusSearch (m_NormalRadius);

		// Compute the features
		ne.compute (*pNormals);
	}

	void computeFPFHDescriptor(
		PointCloudPtr pCloud, 
		NormalCloudPtr pNormals,
		DescriptorPtr fpfhs)
	{
		pcl::FPFHEstimation<PointT, NormalT, SignatureT> fpfh;
		fpfh.setInputCloud (pCloud);
		fpfh.setInputNormals (pNormals);
		// alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);

		// Create an empty kdtree representation, and pass it to the FPFH estimation object.
		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
		pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);

		fpfh.setSearchMethod (tree);

		// Output datasets
		// * initialize outside

		// Use all neighbors in a sphere of radius 5cm
		// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
		fpfh.setRadiusSearch (m_FpfhRadius);

		// Compute the features
		fpfh.compute (*fpfhs);
	}


	//void categorize(std::vector<Cloudject> cloudjects, DescriptorPtr fpfhs, int& category)
	//{
	//	int closerCloudject;

	//	float dist;
	//	float closerCloudjectDist = std::numeric_limits<float>::infinity();

	//	for (int i = 0; i < cloudjects.size(); i++)
	//	{
	//		dist = cloudjects[i].matchView(fpfhs);
	//		if (dist < closerCloudjectDist) 
	//		{ 
	//			closerCloudjectDist = dist;
	//			closerCloudject = i;
	//		}
	//	}

	//	// Final categorization as the closer cloudject in term of distance between point signatures
	//	category = closerCloudject;
	//}


	//void perform(std::vector<PointCloudPtr> views, std::vector<int>& categories)
	//{
	//	// Describe

	//	//std::cout << "Feature extraction in training ..." << std::endl;

	//	for (int v = 0; v < views.size(); v++)
	//	{
	//		NormalCloudPtr pViewNormals (new NormalCloud);

	//		views[v]
	//		pcl::PointCloud<pcl::FPFHSignature33>::Ptr pViewFPFHSignature (new pcl::PointCloud<pcl::FPFHSignature33>);
	//		computeFPFHDescriptor(views[v], pViewNormals, pViewFPFHSignature);

	//		int category;
	//		categorize(m_TrCloudjects, pViewFPFHSignature, category);
	//		//std::cout << category << std::endl;
	//		categories.push_back(category);
	//	}
	//}

private:
	std::vector<CloudjectModel> m_TrModels;
	std::vector<Cloudject> m_TeCloudjects;

	float m_leafSize;
	float m_NormalRadius;
	float m_FpfhRadius;
};


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


	//void downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr view, float leafSize, pcl::PointCloud<pcl::PointXYZ>::Ptr filteredView)
	//{
	//	pcl::ApproximateVoxelGrid<pcl::PointXYZ> sor;
	//	sor.setInputCloud(view);
	//	sor.setLeafSize(leafSize,leafSize,leafSize);
	//	sor.filter(*filteredView);
	//}


	void createCloudjectModels(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> views, int numObjects, int numInstances, 
		float leafSize, std::vector<CloudjectModel>& models)
	{
		for (int i = 0; i < numObjects; i++)
		{
			CloudjectModel model(i, m_NumInstancesTrain, leafSize);
			for (int j = 0; j < numInstances; j++)
			{
				//pcl::PointCloud<pcl::PointXYZ>::Ptr filteredView (new pcl::PointCloud<pcl::PointXYZRGB>);
				int idx = i*numInstances+j;
				//downsample(views[idx], leafSize, filteredView);
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
			float minDist = std::numeric_limits<float>::infinity();

			for (int m = 0; m < models.size(); m++)
			{
				float dist = models[m].match(cloudjects[i]);
				if (dist < minDist)
				{
					categories[i] = m;
					minDist = dist;
				}
			}

			// TODO: Rejection implementation
			// if (minDist > m_MaxCorrespondenceThres)
			// ... assign an arbitrary class ID
		}
	}


	void run()
	{
		std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> trViews, teViews;
		
		std::cout << "Loading training ... " << std::endl;
		loadObjectViews("../data/TrainPCDs/", trViews);

		std::cout << "Loading testing ... " << std::endl;
		loadObjectViews("../data/TestPCDs/", teViews);

		std::cout << "Categorizing ... " << std::endl;
		static const float leafSizes[] = {0.030, 0.020, 0.015, 0.010};//{0.010, 0.015, 0.020, 0.03};
		static const float normalsRadius[] = {0.06, 0.04, 0.03, 0.02, 0.01};//{0.01, 0.03, 0.06};
		static const float fpfhRadius[] = {0.100, 0.075, 0.050, 0.025};// {0.025, 0.05, 0.075, 0.1};

		std::vector<float> vLeafSizes (leafSizes, leafSizes + sizeof(leafSizes) / sizeof(leafSizes[0]) );
		std::vector<float> vNormalsRadius (normalsRadius, normalsRadius + sizeof(normalsRadius) / sizeof(normalsRadius[0]) );
		std::vector<float> vFpfhRadius (fpfhRadius, fpfhRadius + sizeof(fpfhRadius) / sizeof(fpfhRadius[0]) );

		for (int i = 0; i < vLeafSizes.size(); i++)
		{
			// Downsample train
			std::vector<CloudjectModel> cjModels;
			createCloudjectModels(trViews, m_NumObjects, m_NumInstancesTrain, vLeafSizes[i], cjModels);
			// Downsample test
			std::vector<Cloudject> cjs;
			createCloudjects(teViews, vLeafSizes[i], cjs); 

			for (int j = 0; j < vNormalsRadius.size(); j++)
			{
				for (int k = 0; k < vFpfhRadius.size(); k++)
				{
					std::stringstream resultsline;
					resultsline << vLeafSizes[i] << " " << vNormalsRadius[j] << " " << vFpfhRadius[k] << " ";

					std::cout << resultsline.str() << std::endl;

					float accuracy;
					float descriptionTime, categorizationTime;

					if (vNormalsRadius[j] < vLeafSizes[i] || vFpfhRadius[k] < vLeafSizes[i])
					{
						accuracy = 0;
						descriptionTime = 0;
						categorizationTime = 0;
					}
					else
					{
						boost::timer t;
						describeCloudjectModels(cjModels, vNormalsRadius[j], vFpfhRadius[k]);
						describeCloudjects(cjs, vNormalsRadius[j], vFpfhRadius[k]);
						descriptionTime = t.elapsed();

						std::vector<int> categories;
					
						t.restart();
						categorize(cjModels, cjs, categories);
						categorizationTime = t.elapsed();

						accuracy = computeAccuracy(categories);
					}

					resultsline << accuracy << " " << descriptionTime << " " << categorizationTime << "\n";

					FILE* pFile;
					pFile = fopen("../data/results.txt", "a");
					if (pFile != NULL)
					{
						fputs(resultsline.str().c_str(), pFile);
						fclose(pFile);
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