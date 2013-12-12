#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>


template<typename PointT, typename U>
class CloudjectBase
{
	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename PointCloud::Ptr PointCloudPtr;
public:
	CloudjectBase(void) { m_ID = -1; }

	CloudjectBase(PointCloudPtr viewA, float leafSize = 0.0)
	{
		m_ID = -1;
		m_Type = Type::OneView;

		m_OriginalViewA = viewA;

		if (leafSize > 0.0)
		{
			m_ViewA = PointCloudPtr(new PointCloud);

			downsample(m_OriginalViewA, leafSize, m_ViewA);
		}
		else
		{
			m_ViewA = PointCloudPtr(new PointCloud);
				
			pcl::copyPointCloud(*m_OriginalViewA, *m_ViewA);
		}

		// TODO: the position should be the mean of the the two centroids from the two views, not from only A
		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*m_ViewA, centroid);
		m_PosA.x = centroid.x();
		m_PosA.y = centroid.y();
		m_PosA.z = centroid.z();

		m_MedianDistA = medianDistanceToCentroid(m_ViewA, m_PosA);
	}


	CloudjectBase(PointCloudPtr viewA, PointCloudPtr viewB, float leafSize = 0.0)
	{
		m_ID = -1;
		m_Type = Type::TwoViews;

		m_OriginalViewA = viewA;
		m_OriginalViewB = viewB;

		if (leafSize > 0.0)
		{
			m_ViewA = PointCloudPtr(new PointCloud);
			m_ViewB = PointCloudPtr(new PointCloud);

			downsample(m_OriginalViewA, leafSize, m_ViewA);
			downsample(m_OriginalViewB, leafSize, m_ViewB);
		}
		else
		{
			m_ViewA = PointCloudPtr(new PointCloud);
			m_ViewB = PointCloudPtr(new PointCloud);
				
			pcl::copyPointCloud(*m_OriginalViewA, *m_ViewA);
			pcl::copyPointCloud(*m_OriginalViewB, *m_ViewB);
		}

		// Compute centroids' positions

		Eigen::Vector4f centroidA, centroidB;

		pcl::compute3DCentroid(*m_ViewA, centroidA);
		pcl::compute3DCentroid(*m_ViewB, centroidB);

		m_PosA.x = centroidA.x();
		m_PosA.y = centroidA.y();
		m_PosA.z = centroidA.z();

		m_PosB.x = centroidB.x();
		m_PosB.y = centroidB.y();
		m_PosB.z = centroidB.z();

		m_MedianDistA = medianDistanceToCentroid(m_ViewA, m_PosA);
		m_MedianDistB = medianDistanceToCentroid(m_ViewB, m_PosB);
	}


	CloudjectBase(const char* viewPathA, const char* viewPathB, float leafSize = 0.0)
	{
		m_ID = -1;
		m_Type = Type::TwoViews;

		m_OriginalViewA = PointCloudPtr(new PointCloud);
		m_OriginalViewB = PointCloudPtr(new PointCloud);
	
		pcl::PCDReader reader;
		reader.read(viewPathA, *m_OriginalViewA);
		reader.read(viewPathB, *m_OriginalViewB);

		if (leafSize > 0.0)
		{
			m_ViewA = PointCloudPtr(new PointCloud);
			m_ViewB = PointCloudPtr(new PointCloud);

			downsample(m_OriginalViewA, leafSize, m_ViewA);
			downsample(m_OriginalViewA, leafSize, m_ViewB);
		}
		else
		{
			m_ViewA = viewA;
			m_ViewB = viewB;
		}

		// Compute centroids' positions

		Eigen::Vector4f centroidA, centroidB;

		pcl::compute3DCentroid(*m_ViewA, centroidA);
		pcl::compute3DCentroid(*m_ViewB, centroidB);

		m_PosA.x = centroidA.x();
		m_PosA.y = centroidA.y();
		m_PosA.z = centroidA.z();

		m_PosB.x = centroidB.x();
		m_PosB.y = centroidB.y();
		m_PosB.z = centroidB.z();

		m_MedianDistA = medianDistanceToCentroid(m_ViewA, m_PosA);
		m_MedianDistB = medianDistanceToCentroid(m_ViewB, m_PosB);
	}


	CloudjectBase(const CloudjectBase& cloudject)
	{
		m_ID	= cloudject.m_ID;
		m_Type	= cloudject.m_Type;
		m_OriginalViewA = cloudject.m_OriginalViewA;
		m_OriginalViewB = cloudject.m_OriginalViewB;
		m_ViewA = cloudject.m_ViewA;
		m_ViewB = cloudject.m_ViewB;
		m_PosA	= cloudject.m_PosA;
		m_PosB	= cloudject.m_PosB;
		m_MedianDistA = cloudject.m_MedianDistA;
		m_MedianDistB = cloudject.m_MedianDistB;
	}


	virtual ~CloudjectBase(void) {}

	
	int getID()
	{
		return m_ID;
	}


	void setID(int ID)
	{
		m_ID = ID;
	}


	PointT getPosA()
	{
		return m_PosA;
	}


	PointT getPosB()
	{
		return m_PosB;
	}


	int getNumOfPointsInOriginalViewA()
	{
		return m_OriginalViewA->points.size();
	}


	int getNumOfPointsInOriginalViewB()
	{
		return m_OriginalViewB->points.size();
	}


	int getNumOfPointsInViewA()
	{
		return m_ViewA->points.size();
	}


	int getNumOfPointsInViewB()
	{
		return m_ViewB->points.size();
	}


	float medianDistToCentroidInViewA()
	{
		return m_MedianDistA;
	}


	float medianDistToCentroidInViewB()
	{
		return m_MedianDistB;
	}


	int getType()
	{
		return m_Type;
	}


	PointCloudPtr getViewA()
	{
		return m_ViewA;
	}


	PointCloudPtr getViewB()
	{
		return m_ViewB;
	}


	float euclideanDistance(PointT p1, PointT p2)
	{
		return sqrt(powf(p1.x - p2.x, 2) + powf(p1.y - p2.y, 2) + powf(p1.z - p2.z, 2));
	}


	float medianDistanceToCentroid(PointCloudPtr pCloud, PointT centroid)
	{
		std::vector<float> distances;

		distances.push_back(euclideanDistance(centroid, pCloud->points[0]));

		for (int i = 1; i < pCloud->points.size(); i++)
		{
			float dist = euclideanDistance(centroid, pCloud->points[i]);
			bool inserted = false;
			for (int j = 0; j < distances.size() && !inserted; j++)
			{
				if (dist < distances[j])
				{
					distances.insert(distances.begin() + j, dist);
					inserted = true;
				}
			}
		}

		return distances[(int)(distances.size() / 2)];
	}


	enum Type { OneView, TwoViews };


protected:
	// Methods

	void downsample(PointCloudPtr pCloud, float leafSize, PointCloudPtr pFilteredCloud)
	{
		pcl::ApproximateVoxelGrid<PointT> avg;
		avg.setInputCloud(pCloud);
		avg.setLeafSize(leafSize, leafSize, leafSize);
		avg.filter(*pFilteredCloud);
	}

	// Members

	int m_ID;
	int m_Type;
	PointCloudPtr m_OriginalViewA, m_OriginalViewB;
	PointCloudPtr m_ViewA, m_ViewB;
	PointT m_PosA, m_PosB;
	float m_MedianDistA, m_MedianDistB;
};


template<typename PointT, typename U>
class Cloudject : public CloudjectBase<PointT,U>
{
	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT> PointCloudPtr;

public:
	Cloudject() : CloudjectBase<PointT,U>() {}
	Cloudject(PointCloudPtr viewA, float leafSize = 0.0) 
		: CloudjectBase<PointT,U>(viewA, leafSize) {}
	Cloudject(PointCloudPtr viewA, PointCloudPtr viewB, float leafSize = 0.0) 
		: CloudjectBase<PointT,U>(viewA, viewB, leafSize) {}
	Cloudject(const char* viewPathA, const char* viewPathB, float leafSize = 0.0) 
		: CloudjectBase<PointT,U>(viewPathA, viewPathB, leafSize) {}
	Cloudject(const Cloudject<PointT,U>& cloudject) 
		: CloudjectBase<PointT,U>(cloudject) {}

	virtual ~Cloudject() {}

	int getID() { return CloudjectBase<PointT,U>::getID(); }
	void setID(int ID) { CloudjectBase<PointT,U>::setID(ID); }
	PointT getPosA() { return CloudjectBase<PointT,U>::getPosA(); }
	PointT getPosB() { return CloudjectBase<PointT,U>::getPosB(); }
	int getNumOfPointsInOriginalViewA() { return CloudjectBase<PointT,U>::getNumOfPointsInOriginalViewA(); }
	int getNumOfPointsInOriginalViewB() { return CloudjectBase<PointT,U>::getNumOfPointsInOriginalViewB(); }
	int getNumOfPointsInViewA() { return CloudjectBase<PointT,U>::getNumOfPointsInViewA(); }
	int getNumOfPointsInViewB() { return CloudjectBase<PointT,U>::getNumOfPointsInViewB(); }
	float medianDistToCentroidInViewA() { return CloudjectBase<PointT,U>::medianDistToCentroidInViewA(); }
	float medianDistToCentroidInViewB() { return CloudjectBase<PointT,U>::medianDistToCentroidInViewB(); }
	PointCloudPtr getViewA() { return CloudjectBase<PointT,U>::getViewA(); }
	PointCloudPtr getViewB() { return CloudjectBase<PointT,U>::getViewB(); }
	float euclideanDistance(PointT p1, PointT p2) { return euclideanDistance<PointT,U>::euclideanDistance(p1,p2); }
	float medianDistanceToCentroid(PointCloudPtr pCloud, PointT centroid) 
	{ 
		return medianDistanceToCentroid<PointT,U>::euclideanDistance(pCloud, centroid); 
	}

private:
	void downsample(PointCloudPtr pCloud, float leafSize, PointCloudPtr pFilteredCloud)
	{
		CloudjectBase<PointT,U>::downsample(pCloud, leafSize, pFilteredCloud);
	}
};


template<typename PointT>
class Cloudject<PointT, pcl::FPFHSignature33> : public CloudjectBase<PointT, pcl::FPFHSignature33>
{
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;

public:
	Cloudject() 
		: CloudjectBase<PointT,pcl::FPFHSignature33>() { }
	Cloudject(PointCloudPtr viewA, float leafSize = 0.0) 
		: CloudjectBase<PointT,pcl::FPFHSignature33>(viewA, leafSize) { }
	Cloudject(PointCloudPtr viewA, PointCloudPtr viewB, float leafSize = 0.0) 
		: CloudjectBase<PointT,pcl::FPFHSignature33>(viewA, viewB, leafSize) { }
	Cloudject(const char* viewPathA, const char* viewPathB, float leafSize = 0.0) 
		: CloudjectBase<PointT,pcl::FPFHSignature33>(viewPathA, viewPathB, leafSize) { }
	

	Cloudject(const Cloudject<PointT,pcl::FPFHSignature33>& cloudject) 
		: CloudjectBase<PointT,pcl::FPFHSignature33>(cloudject)
	{
		m_DescriptorA = cloudject.m_DescriptorA;
		m_DescriptorB = cloudject.m_DescriptorB;
	}

	virtual ~Cloudject() {}


	int getID() { return CloudjectBase<PointT,pcl::FPFHSignature33>::getID(); }
	void setID(int ID) { CloudjectBase<PointT,pcl::FPFHSignature33>::setID(ID); }

	PointT getPosA() { return CloudjectBase<PointT,pcl::FPFHSignature33>::getPosA(); }
	PointT getPosB() { return CloudjectBase<PointT,pcl::FPFHSignature33>::getPosB(); }

	int getNumOfPointsInOriginalViewA() { return CloudjectBase<PointT,pcl::FPFHSignature33>::getNumOfPointsInOriginalViewA(); }
	int getNumOfPointsInOriginalViewB() { return CloudjectBase<PointT,pcl::FPFHSignature33>::getNumOfPointsInOriginalViewB(); }
	int getNumOfPointsInViewA() { return CloudjectBase<PointT,pcl::FPFHSignature33>::getNumOfPointsInViewA(); }
	int getNumOfPointsInViewB() { return CloudjectBase<PointT,pcl::FPFHSignature33>::getNumOfPointsInViewB(); }

	float medianDistToCentroidInViewA() { return CloudjectBase<PointT,pcl::FPFHSignature33>::medianDistToCentroidInViewA(); }
	float medianDistToCentroidInViewB() { return CloudjectBase<PointT,pcl::FPFHSignature33>::medianDistToCentroidInViewB(); }

	PointCloudPtr getViewA() { return CloudjectBase<PointT,pcl::FPFHSignature33>::getViewA(); }
	PointCloudPtr getViewB() { return CloudjectBase<PointT,pcl::FPFHSignature33>::getViewB(); }

	float euclideanDistance(PointT p1, PointT p2) { return euclideanDistance<PointT,pcl::FPFHSignature33>::euclideanDistance(p1,p2); }
	float medianDistanceToCentroid(PointCloudPtr pCloud, PointT centroid) 
	{ 
		return medianDistanceToCentroid<PointT,pcl::FPFHSignature33>::euclideanDistance(pCloud, centroid); 
	}


	void init()
	{
		m_DescriptorA = pcl::PointCloud<pcl::FPFHSignature33>::Ptr (new pcl::PointCloud<pcl::FPFHSignature33>);
		m_DescriptorB = pcl::PointCloud<pcl::FPFHSignature33>::Ptr (new pcl::PointCloud<pcl::FPFHSignature33>);
	}


	void describe(pcl::PointCloud<pcl::FPFHSignature33>::Ptr descA, pcl::PointCloud<pcl::FPFHSignature33>::Ptr descB)
	{
		m_DescriptorA = descA;
		m_DescriptorB = descB;
	}

	void describe(float normalRadius, float fpfhRadius)
	{
		m_DescriptorA = pcl::PointCloud<pcl::FPFHSignature33>::Ptr (new pcl::PointCloud<pcl::FPFHSignature33>);
		describeView(m_ViewA, normalRadius, fpfhRadius, *m_DescriptorA);

		if (getType() == Type::TwoViews)
		{
			m_DescriptorB = pcl::PointCloud<pcl::FPFHSignature33>::Ptr (new pcl::PointCloud<pcl::FPFHSignature33>);
			describeView(m_ViewB, normalRadius, fpfhRadius, *m_DescriptorB);
		}
	}


	void describeView(PointCloudPtr pView, 
					  float normalRadius, float fpfhRadius,
					  pcl::PointCloud<pcl::FPFHSignature33>& descriptor)
	{
		//
		// Normals preprocess
		//

		// Create the normal estimation class, and pass the input dataset to it
		pcl::NormalEstimation<PointT, pcl::Normal> ne;
		ne.setInputCloud (pView);

		// Create an empty kdtree representation, and pass it to the normal estimation object.
		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
		pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
		ne.setSearchMethod (tree);

		// Output datasets
		pcl::PointCloud<pcl::Normal>::Ptr pNormals (new pcl::PointCloud<pcl::Normal>);

		// Use all neighbors in a sphere of radius 3cm
		ne.setRadiusSearch (normalRadius);

		// Compute the features
		ne.compute (*pNormals);	

		//
		// FPFH description extraction
		//

		pcl::FPFHEstimationOMP<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
		fpfh.setInputCloud (pView);
		fpfh.setInputNormals (pNormals);
		// alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);

		// Create an empty kdtree representation, and pass it to the FPFH estimation object.
		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
		tree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>);
		fpfh.setSearchMethod (tree);

		// Output datasets
		// * initialize outside

		// Use all neighbors in a sphere of radius 5cm
		// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
		fpfh.setRadiusSearch (fpfhRadius);

		// Compute the features
		fpfh.compute (descriptor);
	}


	pcl::PointCloud<pcl::FPFHSignature33>::Ptr getDescriptionA()
	{
		return m_DescriptorA;
	}


	pcl::PointCloud<pcl::FPFHSignature33>::Ptr getDescriptionB()
	{
		return m_DescriptorB;
	}


private:
	void downsample(PointCloudPtr pCloud, float leafSize, PointCloudPtr pFilteredCloud)
	{
		CloudjectBase<PointT,pcl::FPFHSignature33>::downsample(pCloud, leafSize, pFilteredCloud);
	}

	pcl::PointCloud<pcl::FPFHSignature33>::Ptr m_DescriptorA, m_DescriptorB;
};


template<typename PointT>
class Cloudject<PointT, pcl::PFHRGBSignature250> : public CloudjectBase<PointT, pcl::PFHRGBSignature250>
{
	typedef pcl::PFHRGBSignature250 SignatureT;
	typedef pcl::PointCloud<SignatureT> Descriptor;
	typedef pcl::PointCloud<SignatureT>::Ptr DescriptorPtr;
	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;

public:
	Cloudject() 
		: CloudjectBase<PointT, SignatureT>() { }
	Cloudject(PointCloudPtr viewA, float leafSize = 0.0) 
		: CloudjectBase<PointT, SignatureT>(viewA, leafSize) { }
	Cloudject(PointCloudPtr viewA, PointCloudPtr viewB, float leafSize = 0.0) 
		: CloudjectBase<PointT, SignatureT>(viewA, viewB, leafSize) { }
	Cloudject(const char* viewPathA, const char* viewPathB, float leafSize = 0.0) 
		: CloudjectBase<PointT, SignatureT>(viewPathA, viewPathB, leafSize) { }
	

	Cloudject(const Cloudject<PointT, SignatureT>& cloudject) 
		: CloudjectBase<PointT, SignatureT>(cloudject)
	{
		m_DescriptorA = cloudject.m_DescriptorA;
		m_DescriptorB = cloudject.m_DescriptorB;
	}

	virtual ~Cloudject() {}


	int getID() { return CloudjectBase<PointT, SignatureT>::getID(); }
	
	void setID(int ID) { CloudjectBase<PointT, SignatureT>::setID(ID); }


	PointT getPosA() { return CloudjectBase<PointT, SignatureT>::getPosA(); }

	PointT getPosB() { return CloudjectBase<PointT, SignatureT>::getPosB(); }


	PointCloudPtr getViewA() { return CloudjectBase<PointT, SignatureT>::getViewA(); }

	PointCloudPtr getViewB() { return CloudjectBase<PointT, SignatureT>::getViewB(); }


	void init()
	{
		m_DescriptorA = DescriptorPtr (new Descriptor);
		m_DescriptorB = DescriptorPtr (new Descriptor);
	}

	void describe(DescriptorPtr descA, DescriptorPtr descB)
	{
		m_DescriptorA = descA;
		m_DescriptorB = descB;
	}

	void describe(float normalRadius, float fpfhRadius)
	{
		m_DescriptorA = DescriptorPtr (new Descriptor);
		describeView(m_ViewA, normalRadius, fpfhRadius, *m_DescriptorA);

		if (getType() == Type::TwoViews)
		{
			m_DescriptorB = DescriptorPtr (new Descriptor);
			describeView(m_ViewB, normalRadius, fpfhRadius, *m_DescriptorB);
		}
	}

	void describeView(PointCloudPtr pView, 
					  float normalRadius, float featureRadius,
					  Descriptor& descriptor)
	{
		//
		// Normals preprocess
		//

		// Create the normal estimation class, and pass the input dataset to it
		pcl::NormalEstimation<PointT, pcl::Normal> ne;
		ne.setInputCloud (pView);

		// Create an empty kdtree representation, and pass it to the normal estimation object.
		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
		pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
		ne.setSearchMethod (tree);

		// Output datasets
		pcl::PointCloud<pcl::Normal>::Ptr pNormals (new pcl::PointCloud<pcl::Normal>);

		// Use all neighbors in a sphere of radius 3cm
		ne.setRadiusSearch (normalRadius);

		// Compute the features
		ne.compute (*pNormals);	

		//
		// FPFH description extraction
		//

		pcl::PFHRGBEstimation<PointT, pcl::Normal, SignatureT> pfhrgb;
		pfhrgb.setInputCloud (pView);
		pfhrgb.setInputNormals (pNormals);
		// alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);

		// Create an empty kdtree representation, and pass it to the FPFH estimation object.
		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
		tree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>);
		pfhrgb.setSearchMethod (tree);

		// Output datasets
		// * initialize outside

		// Use all neighbors in a sphere of radius 5cm
		// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
		pfhrgb.setRadiusSearch (featureRadius);

		// Compute the features
		pfhrgb.compute (descriptor);
	}


	DescriptorPtr getDescriptionA()
	{
		return m_DescriptorA;
	}


	DescriptorPtr getDescriptionB()
	{
		return m_DescriptorB;
	}


private:
	void downsample(PointCloudPtr pCloud, float leafSize, PointCloudPtr pFilteredCloud)
	{
		CloudjectBase<PointT,SignatureT>::downsample(pCloud, leafSize, pFilteredCloud);
	}

	DescriptorPtr m_DescriptorA, m_DescriptorB;
};

