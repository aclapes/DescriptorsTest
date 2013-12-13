#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/features/normal_3d.h>


template<typename PointT, typename SignatureT>
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


// Locally Featured (LF) Cloudject
template<typename PointT, typename SignatureT>
class LFCloudjectBase : public CloudjectBase<PointT,SignatureT>
{
	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
	typedef typename pcl::PointCloud<SignatureT> Descriptor;
	typedef typename pcl::PointCloud<SignatureT>::Ptr DescriptorPtr;

public:
	LFCloudjectBase() : CloudjectBase<PointT,SignatureT>() {}

	LFCloudjectBase(PointCloudPtr viewA, float leafSize = 0.0) 
		: CloudjectBase<PointT,SignatureT>(viewA, leafSize) {}

	LFCloudjectBase(PointCloudPtr viewA, PointCloudPtr viewB, float leafSize = 0.0) 
		: CloudjectBase<PointT,SignatureT>(viewA, viewB, leafSize) {}

	LFCloudjectBase(const char* viewPathA, const char* viewPathB, float leafSize = 0.0) 
		: CloudjectBase<PointT,SignatureT>(viewPathA, viewPathB, leafSize) {}

	LFCloudjectBase(const LFCloudjectBase<PointT,SignatureT>& cloudject) 
		: CloudjectBase<PointT,SignatureT>(cloudject) 
	{
		m_DescriptorA = cloudject.m_DescriptorA;
		m_DescriptorB = cloudject.m_DescriptorB;
	}

	virtual ~LFCloudjectBase() {}


	int getID() { return CloudjectBase<PointT,SignatureT>::getID(); }
	void setID(int ID) { CloudjectBase<PointT,SignatureT>::setID(ID); }

	PointT getPosA() { return CloudjectBase<PointT,SignatureT>::getPosA(); }
	PointT getPosB() { return CloudjectBase<PointT,SignatureT>::getPosB(); }

	int getNumOfPointsInOriginalViewA() { return CloudjectBase<PointT,SignatureT>::getNumOfPointsInOriginalViewA(); }
	int getNumOfPointsInOriginalViewB() { return CloudjectBase<PointT,SignatureT>::getNumOfPointsInOriginalViewB(); }

	int getNumOfPointsInViewA() { return CloudjectBase<PointT,SignatureT>::getNumOfPointsInViewA(); }
	int getNumOfPointsInViewB() { return CloudjectBase<PointT,SignatureT>::getNumOfPointsInViewB(); }

	float medianDistToCentroidInViewA() { return CloudjectBase<PointT,SignatureT>::medianDistToCentroidInViewA(); }
	float medianDistToCentroidInViewB() { return CloudjectBase<PointT,SignatureT>::medianDistToCentroidInViewB(); }

	PointCloudPtr getViewA() { return CloudjectBase<PointT,SignatureT>::getViewA(); }
	PointCloudPtr getViewB() { return CloudjectBase<PointT,SignatureT>::getViewB(); }

	float euclideanDistance(PointT p1, PointT p2) { return CloudjectBase<PointT,SignatureT>::euclideanDistance(p1,p2); }
	float medianDistanceToCentroid(PointCloudPtr pCloud, PointT centroid) 
	{ return CloudjectBase<PointT,SignatureT>::euclideanDistance(pCloud, centroid); }


	void describe(DescriptorPtr descA, DescriptorPtr descB)
	{
		m_DescriptorA = descA;
		m_DescriptorB = descB;
	}


	DescriptorPtr getDescriptionA()
	{ 
		return m_DescriptorA;
	}


	DescriptorPtr getDescriptionB()
	{ 
		return m_DescriptorB;
	}


protected:
	DescriptorPtr m_DescriptorA, m_DescriptorB;

private:
	//void downsample(PointCloudPtr pCloud, float leafSize, PointCloudPtr pFilteredCloud)
	//{ CloudjectBase<PointT,SignatureT>::downsample(pCloud, leafSize, pFilteredCloud); }
};


// Locally Featured (LF) Cloudject
template<typename PointT, typename SignatureT>
class LFCloudject : public LFCloudjectBase<PointT,SignatureT> { };


template<typename PointT>
class LFCloudject<typename PointT, pcl::FPFHSignature33> : public LFCloudjectBase<PointT, pcl::FPFHSignature33>
{ 
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;

public:
	LFCloudject() 
		: LFCloudjectBase<PointT,pcl::FPFHSignature33>() { }

	LFCloudject(PointCloudPtr viewA, float leafSize = 0.0) 
		: LFCloudjectBase<PointT,pcl::FPFHSignature33>(viewA, leafSize) { }

	LFCloudject(PointCloudPtr viewA, PointCloudPtr viewB, float leafSize = 0.0) 
		: LFCloudjectBase<PointT,pcl::FPFHSignature33>(viewA, viewB, leafSize) { }

	LFCloudject(const char* viewPathA, const char* viewPathB, float leafSize = 0.0) 
		: LFCloudjectBase<PointT,pcl::FPFHSignature33>(viewPathA, viewPathB, leafSize) { }

	LFCloudject(const LFCloudject<PointT,pcl::FPFHSignature33>& cloudject) 
		: LFCloudjectBase<PointT,pcl::FPFHSignature33>(cloudject) { }

	virtual ~LFCloudject() {}

	//
	// Methods
	// 
	
	int getID() { return LFCloudjectBase<PointT,pcl::FPFHSignature33>::getID(); }
	void setID(int ID) { LFCloudjectBase<PointT,pcl::FPFHSignature33>::setID(ID); }

	PointT getPosA() { return LFCloudjectBase<PointT,pcl::FPFHSignature33>::getPosA(); }
	PointT getPosB() { return LFCloudjectBase<PointT,pcl::FPFHSignature33>::getPosB(); }

	int getNumOfPointsInOriginalViewA() { return LFCloudjectBase<PointT,pcl::FPFHSignature33>::getNumOfPointsInOriginalViewA(); }
	int getNumOfPointsInOriginalViewB() { return LFCloudjectBase<PointT,pcl::FPFHSignature33>::getNumOfPointsInOriginalViewB(); }
	int getNumOfPointsInViewA() { return LFCloudjectBase<PointT,pcl::FPFHSignature33>::getNumOfPointsInViewA(); }
	int getNumOfPointsInViewB() { return LFCloudjectBase<PointT,pcl::FPFHSignature33>::getNumOfPointsInViewB(); }

	float medianDistToCentroidInViewA() { return LFCloudjectBase<PointT,pcl::FPFHSignature33>::medianDistToCentroidInViewA(); }
	float medianDistToCentroidInViewB() { return LFCloudjectBase<PointT,pcl::FPFHSignature33>::medianDistToCentroidInViewB(); }

	PointCloudPtr getViewA() { return LFCloudjectBase<PointT,pcl::FPFHSignature33>::getViewA(); }
	PointCloudPtr getViewB() { return LFCloudjectBase<PointT,pcl::FPFHSignature33>::getViewB(); }

	float euclideanDistance(PointT p1, PointT p2) { return LFCloudjectBase<PointT,pcl::FPFHSignature33>::euclideanDistance(p1,p2); }
	float medianDistanceToCentroid(PointCloudPtr pCloud, PointT centroid) 
	{ return LFCloudjectBase<PointT,pcl::FPFHSignature33>::euclideanDistance(pCloud, centroid); }

	void describe(pcl::PointCloud<pcl::FPFHSignature33>::Ptr descA, pcl::PointCloud<pcl::FPFHSignature33>::Ptr descB)
	{ LFCloudjectBase<PointT,pcl::FPFHSignature33>::describe(descA, descB); }

	pcl::PointCloud<pcl::FPFHSignature33>::Ptr getDescriptionA()
	{ return LFCloudjectBase<PointT,pcl::FPFHSignature33>::getDescriptionA(); }
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr getDescriptionB()
	{ return LFCloudjectBase<PointT,pcl::FPFHSignature33>::getDescriptionB(); }


	void describe(float normalRadius, float fpfhRadius)
	{
		m_DescriptorA = pcl::PointCloud<pcl::FPFHSignature33>::Ptr (new  pcl::PointCloud<pcl::FPFHSignature33>);
		describeView(m_ViewA, normalRadius, fpfhRadius, *m_DescriptorA);

		if (getType() == Type::TwoViews)
		{
			m_DescriptorB = pcl::PointCloud<pcl::FPFHSignature33>::Ptr (new  pcl::PointCloud<pcl::FPFHSignature33>);
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

		pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
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

private:
	void downsample(PointCloudPtr pCloud, float leafSize, PointCloudPtr pFilteredCloud)
	{ LFCloudjectBase<PointT,pcl::FPFHSignature33>::downsample(pCloud, leafSize, pFilteredCloud); }
};


template<typename PointT>
class LFCloudject<typename PointT, pcl::PFHRGBSignature250> : public LFCloudjectBase<PointT, pcl::PFHRGBSignature250>
{ 
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;

public:
	LFCloudject() 
		: LFCloudjectBase<PointT,pcl::PFHRGBSignature250>() { }

	LFCloudject(PointCloudPtr viewA, float leafSize = 0.0) 
		: LFCloudjectBase<PointT,pcl::PFHRGBSignature250>(viewA, leafSize) { }

	LFCloudject(PointCloudPtr viewA, PointCloudPtr viewB, float leafSize = 0.0) 
		: LFCloudjectBase<PointT,pcl::PFHRGBSignature250>(viewA, viewB, leafSize) { }

	LFCloudject(const char* viewPathA, const char* viewPathB, float leafSize = 0.0) 
		: LFCloudjectBase<PointT,pcl::PFHRGBSignature250>(viewPathA, viewPathB, leafSize) { }

	LFCloudject(const LFCloudject<PointT,pcl::PFHRGBSignature250>& cloudject) 
		: LFCloudjectBase<PointT,pcl::PFHRGBSignature250>(cloudject) { }

	virtual ~LFCloudject() {}

	//
	// Methods
	// 
	
	int getID() { return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::getID(); }
	void setID(int ID) { LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::setID(ID); }

	PointT getPosA() { return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::getPosA(); }
	PointT getPosB() { return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::getPosB(); }

	int getNumOfPointsInOriginalViewA() { return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::getNumOfPointsInOriginalViewA(); }
	int getNumOfPointsInOriginalViewB() { return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::getNumOfPointsInOriginalViewB(); }
	int getNumOfPointsInViewA() { return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::getNumOfPointsInViewA(); }
	int getNumOfPointsInViewB() { return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::getNumOfPointsInViewB(); }

	float medianDistToCentroidInViewA() { return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::medianDistToCentroidInViewA(); }
	float medianDistToCentroidInViewB() { return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::medianDistToCentroidInViewB(); }

	PointCloudPtr getViewA() { return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::getViewA(); }
	PointCloudPtr getViewB() { return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::getViewB(); }

	float euclideanDistance(PointT p1, PointT p2) { return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::euclideanDistance(p1,p2); }
	float medianDistanceToCentroid(PointCloudPtr pCloud, PointT centroid) 
	{ return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::euclideanDistance(pCloud, centroid); }

	void describe(pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr descA, pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr descB)
	{ LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::describe(descA, descB); }

	pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr getDescriptionA()
	{ return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::getDescriptionA(); }
	pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr getDescriptionB()
	{ return LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::getDescriptionB(); }


	void describe(float normalRadius, float pfhrgbRadius)
	{
		m_DescriptorA = pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr (new  pcl::PointCloud<pcl::PFHRGBSignature250>);
		describeView(m_ViewA, normalRadius, pfhrgbRadius, *m_DescriptorA);

		if (getType() == Type::TwoViews)
		{
			m_DescriptorB = pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr (new  pcl::PointCloud<pcl::PFHRGBSignature250>);
			describeView(m_ViewB, normalRadius, pfhrgbRadius, *m_DescriptorB);
		}
	}


	void describeView(PointCloudPtr pView, 
					  float normalRadius, float pfhrgbRadius,
					  pcl::PointCloud<pcl::PFHRGBSignature250>& descriptor)
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

		pcl::PFHRGBEstimation<PointT, pcl::Normal, pcl::PFHRGBSignature250> pfhrgb;
		pfhrgb.setInputCloud (pView);
		pfhrgb.setInputNormals (pNormals);
		// alternatively, if cloud is of tpe PointNormal, do pfhrgb.setInputNormals (cloud);

		// Create an empty kdtree representation, and pass it to the FPFH estimation object.
		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
		tree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>);
		pfhrgb.setSearchMethod (tree);

		// Output datasets
		// * initialize outside

		// Use all neighbors in a sphere of radius 5cm
		// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
		pfhrgb.setRadiusSearch (pfhrgbRadius);

		// Compute the features
		pfhrgb.compute (descriptor);
	}

private:
	void downsample(PointCloudPtr pCloud, float leafSize, PointCloudPtr pFilteredCloud)
	{ LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::downsample(pCloud, leafSize, pFilteredCloud); }
};


//template<typename PointT>
//class Cloudject<PointT, pcl::FPFHSignature33> : public LFCloudject<PointT, pcl::FPFHSignature33>
//{
//	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
//
//public:
//
//	Cloudject() 
//		: LFCloudject<PointT,pcl::FPFHSignature33>() { }
//
//	Cloudject(PointCloudPtr viewA, float leafSize = 0.0) 
//		: LFCloudject<PointT,pcl::FPFHSignature33>(viewA, leafSize) { }
//
//	Cloudject(PointCloudPtr viewA, PointCloudPtr viewB, float leafSize = 0.0) 
//		: LFCloudject<PointT,pcl::FPFHSignature33>(viewA, viewB, leafSize) { }
//
//	Cloudject(const char* viewPathA, const char* viewPathB, float leafSize = 0.0) 
//		: LFCloudject<PointT,pcl::FPFHSignature33>(viewPathA, viewPathB, leafSize) { }
//
//	Cloudject(const Cloudject<PointT,pcl::FPFHSignature33>& cloudject) 
//		: LFCloudject<PointT,pcl::FPFHSignature33>(cloudject) { }
//
//	virtual ~Cloudject() {}
//
//	//
//	// Methods
//	// 
//	
//	int getID() { return LFCloudject<PointT,pcl::FPFHSignature33>::getID(); }
//	void setID(int ID) { LFCloudject<PointT,pcl::FPFHSignature33>::setID(ID); }
//
//	PointT getPosA() { return LFCloudject<PointT,pcl::FPFHSignature33>::getPosA(); }
//	PointT getPosB() { return LFCloudject<PointT,pcl::FPFHSignature33>::getPosB(); }
//
//	int getNumOfPointsInOriginalViewA() { return LFCloudject<PointT,pcl::FPFHSignature33>::getNumOfPointsInOriginalViewA(); }
//	int getNumOfPointsInOriginalViewB() { return LFCloudject<PointT,pcl::FPFHSignature33>::getNumOfPointsInOriginalViewB(); }
//	int getNumOfPointsInViewA() { return LFCloudject<PointT,pcl::FPFHSignature33>::getNumOfPointsInViewA(); }
//	int getNumOfPointsInViewB() { return LFCloudject<PointT,pcl::FPFHSignature33>::getNumOfPointsInViewB(); }
//
//	float medianDistToCentroidInViewA() { return LFCloudject<PointT,pcl::FPFHSignature33>::medianDistToCentroidInViewA(); }
//	float medianDistToCentroidInViewB() { return LFCloudject<PointT,pcl::FPFHSignature33>::medianDistToCentroidInViewB(); }
//
//	PointCloudPtr getViewA() { return LFCloudject<PointT,pcl::FPFHSignature33>::getViewA(); }
//	PointCloudPtr getViewB() { return LFCloudject<PointT,pcl::FPFHSignature33>::getViewB(); }
//
//	float euclideanDistance(PointT p1, PointT p2) { return LFCloudject<PointT,pcl::FPFHSignature33>::euclideanDistance(p1,p2); }
//	float medianDistanceToCentroid(PointCloudPtr pCloud, PointT centroid) 
//	{ return LFCloudject<PointT,pcl::FPFHSignature33>::euclideanDistance(pCloud, centroid); }
//
//
//	void describe(pcl::PointCloud<pcl::FPFHSignature33>::Ptr descA, pcl::PointCloud<pcl::FPFHSignature33>::Ptr descB)
//	{
//		
//		m_DescriptorA = descA;
//		m_DescriptorB = descB;
//	}
//
//	void describe(float normalRadius, float fpfhRadius)
//	{
//		m_DescriptorA = pcl::PointCloud<pcl::FPFHSignature33>::Ptr (new pcl::PointCloud<pcl::FPFHSignature33>);
//		describeView(m_ViewA, normalRadius, fpfhRadius, *m_DescriptorA);
//
//		if (getType() == Type::TwoViews)
//		{
//			m_DescriptorB = pcl::PointCloud<pcl::FPFHSignature33>::Ptr (new pcl::PointCloud<pcl::FPFHSignature33>);
//			describeView(m_ViewB, normalRadius, fpfhRadius, *m_DescriptorB);
//		}
//	}
//
//
//	void describeView(PointCloudPtr pView, 
//					  float normalRadius, float fpfhRadius,
//					  pcl::PointCloud<pcl::FPFHSignature33>& descriptor)
//	{
//		//
//		// Normals preprocess
//		//
//
//		// Create the normal estimation class, and pass the input dataset to it
//		pcl::NormalEstimation<PointT, pcl::Normal> ne;
//		ne.setInputCloud (pView);
//
//		// Create an empty kdtree representation, and pass it to the normal estimation object.
//		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
//		pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
//		ne.setSearchMethod (tree);
//
//		// Output datasets
//		pcl::PointCloud<pcl::Normal>::Ptr pNormals (new pcl::PointCloud<pcl::Normal>);
//
//		// Use all neighbors in a sphere of radius 3cm
//		ne.setRadiusSearch (normalRadius);
//
//		// Compute the features
//		ne.compute (*pNormals);	
//
//		//
//		// FPFH description extraction
//		//
//
//		pcl::FPFHEstimationOMP<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
//		fpfh.setInputCloud (pView);
//		fpfh.setInputNormals (pNormals);
//		// alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);
//
//		// Create an empty kdtree representation, and pass it to the FPFH estimation object.
//		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
//		tree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>);
//		fpfh.setSearchMethod (tree);
//
//		// Output datasets
//		// * initialize outside
//
//		// Use all neighbors in a sphere of radius 5cm
//		// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
//		fpfh.setRadiusSearch (fpfhRadius);
//
//		// Compute the features
//		fpfh.compute (descriptor);
//	}
//
//
//	pcl::PointCloud<pcl::FPFHSignature33>::Ptr getDescriptionA()
//	{
//		return m_DescriptorA;
//	}
//
//
//	pcl::PointCloud<pcl::FPFHSignature33>::Ptr getDescriptionB()
//	{
//		return m_DescriptorB;
//	}
//
//
//private:
//	void downsample(PointCloudPtr pCloud, float leafSize, PointCloudPtr pFilteredCloud)
//	{
//		CloudjectBase<PointT,pcl::FPFHSignature33>::downsample(pCloud, leafSize, pFilteredCloud);
//	}
//
//
//};


//template<typename PointT>
//class LFCloudject<PointT, pcl::PFHRGBSignature250> : public LFCloudject<PointT, pcl::PFHRGBSignature250>
//{
//	typedef pcl::PFHRGBSignature250 SignatureT;
//	typedef pcl::PointCloud<SignatureT> Descriptor;
//	typedef pcl::PointCloud<SignatureT>::Ptr DescriptorPtr;
//	typedef typename pcl::PointCloud<PointT> PointCloud;
//	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
//
//public:
//	LFCloudject() 
//		: LFCloudject<PointT, SignatureT>() { }
//	LFCloudject(PointCloudPtr viewA, float leafSize = 0.0) 
//		: LFCloudject<PointT, SignatureT>(viewA, leafSize) { }
//	LFCloudject(PointCloudPtr viewA, PointCloudPtr viewB, float leafSize = 0.0) 
//		: LFCloudject<PointT, SignatureT>(viewA, viewB, leafSize) { }
//	LFCloudject(const char* viewPathA, const char* viewPathB, float leafSize = 0.0) 
//		: LFCloudject<PointT, SignatureT>(viewPathA, viewPathB, leafSize) { }
//	
//
//	LFCloudject(const LFCloudject<PointT, SignatureT>& cloudject) 
//		: LFCloudject<PointT, SignatureT>(cloudject)
//	{
//		m_DescriptorA = cloudject.m_DescriptorA;
//		m_DescriptorB = cloudject.m_DescriptorB;
//	}
//
//	virtual ~Cloudject() {}
//
//
//	//int getID() { return LFCloudject<PointT, SignatureT>::getID(); }
//	//
//	//void setID(int ID) { LFCloudject<PointT, SignatureT>::setID(ID); }
//
//
//	//PointT getPosA() { return LFCloudject<PointT, SignatureT>::getPosA(); }
//
//	//PointT getPosB() { return LFCloudject<PointT, SignatureT>::getPosB(); }
//
//
//	//PointCloudPtr getViewA() { return LFCloudject<PointT, SignatureT>::getViewA(); }
//
//	//PointCloudPtr getViewB() { return LFCloudject<PointT, SignatureT>::getViewB(); }
//
//
//	//void describe(DescriptorPtr descA, DescriptorPtr descB)
//	//{
//	//	m_DescriptorA = descA;
//	//	m_DescriptorB = descB;
//	//}
//
//	//void describe(float normalRadius, float fpfhRadius)
//	//{
//	//	m_DescriptorA = DescriptorPtr (new Descriptor);
//	//	describeView(m_ViewA, normalRadius, fpfhRadius, *m_DescriptorA);
//
//	//	if (getType() == Type::TwoViews)
//	//	{
//	//		m_DescriptorB = DescriptorPtr (new Descriptor);
//	//		describeView(m_ViewB, normalRadius, fpfhRadius, *m_DescriptorB);
//	//	}
//	//}
//
//	//void describeView(PointCloudPtr pView, 
//	//				  float normalRadius, float featureRadius,
//	//				  Descriptor& descriptor)
//	//{
//	//	//
//	//	// Normals preprocess
//	//	//
//
//	//	// Create the normal estimation class, and pass the input dataset to it
//	//	pcl::NormalEstimation<PointT, pcl::Normal> ne;
//	//	ne.setInputCloud (pView);
//
//	//	// Create an empty kdtree representation, and pass it to the normal estimation object.
//	//	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
//	//	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
//	//	ne.setSearchMethod (tree);
//
//	//	// Output datasets
//	//	pcl::PointCloud<pcl::Normal>::Ptr pNormals (new pcl::PointCloud<pcl::Normal>);
//
//	//	// Use all neighbors in a sphere of radius 3cm
//	//	ne.setRadiusSearch (normalRadius);
//
//	//	// Compute the features
//	//	ne.compute (*pNormals);	
//
//	//	//
//	//	// FPFH description extraction
//	//	//
//
//	//	pcl::PFHRGBEstimation<PointT, pcl::Normal, SignatureT> pfhrgb;
//	//	pfhrgb.setInputCloud (pView);
//	//	pfhrgb.setInputNormals (pNormals);
//	//	// alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);
//
//	//	// Create an empty kdtree representation, and pass it to the FPFH estimation object.
//	//	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
//	//	tree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>);
//	//	pfhrgb.setSearchMethod (tree);
//
//	//	// Output datasets
//	//	// * initialize outside
//
//	//	// Use all neighbors in a sphere of radius 5cm
//	//	// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
//	//	pfhrgb.setRadiusSearch (featureRadius);
//
//	//	// Compute the features
//	//	pfhrgb.compute (descriptor);
//	//}
//
//
//	//DescriptorPtr getDescriptionA()
//	//{
//	//	return m_DescriptorA;
//	//}
//
//
//	//DescriptorPtr getDescriptionB()
//	//{
//	//	return m_DescriptorB;
//	//}
//
//
//private:
//	//void downsample(PointCloudPtr pCloud, float leafSize, PointCloudPtr pFilteredCloud)
//	//{
//	//	CloudjectBase<PointT,SignatureT>::downsample(pCloud, leafSize, pFilteredCloud);
//	//}
//
//	//DescriptorPtr m_DescriptorA, m_DescriptorB;
//};
//
