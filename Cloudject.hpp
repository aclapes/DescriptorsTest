#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>
#include <pcl/features/fpfh.h>
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

		if (leafSize > 0.0)
		{
			m_ViewA = PointCloudPtr(new PointCloud);

			downsample(viewA, leafSize, m_ViewA);
		}
		else
		{
			m_ViewA = PointCloudPtr(new PointCloud);
				
			pcl::copyPointCloud(*viewA, *m_ViewA);
		}

		// TODO: the position should be the mean of the the two centroids from the two views, not from only A
		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*m_ViewA, centroid);
		m_PosA.x = centroid.x();
		m_PosA.y = centroid.y();
		m_PosA.z = centroid.z();
	}


	CloudjectBase(PointCloudPtr viewA, PointCloudPtr viewB, float leafSize = 0.0)
	{
		m_ID = -1;
		m_Type = Type::TwoViews;

		if (leafSize > 0.0)
		{
			m_ViewA = PointCloudPtr(new PointCloud);
			m_ViewB = PointCloudPtr(new PointCloud);

			downsample(viewA, leafSize, m_ViewA);
			downsample(viewB, leafSize, m_ViewB);
		}
		else
		{
			m_ViewA = PointCloudPtr(new PointCloud);
			m_ViewB = PointCloudPtr(new PointCloud);
				
			pcl::copyPointCloud(*viewA, *m_ViewA);
			pcl::copyPointCloud(*viewB, *m_ViewB);
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
	}


	CloudjectBase(const char* viewPathA, const char* viewPathB, float leafSize = 0.0)
	{
		m_ID = -1;
		m_Type = Type::TwoViews;

		PointCloudPtr viewA (new PointCloud);
		PointCloudPtr viewB (new PointCloud);
	
		pcl::PCDReader reader;
		reader.read(viewPathA, *viewA);
		reader.read(viewPathB, *viewB);

		if (leafSize > 0.0)
		{
			m_ViewA = PointCloudPtr(new PointCloud);
			m_ViewB = PointCloudPtr(new PointCloud);

			downsample(viewA, leafSize, m_ViewA);
			downsample(viewB, leafSize, m_ViewB);
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
	}


	CloudjectBase(const CloudjectBase& cloudject)
	{
		m_ID	= cloudject.m_ID;
		m_Type	= cloudject.m_Type;
		m_ViewA = cloudject.m_ViewA;
		m_ViewB = cloudject.m_ViewB;
		m_PosA	= cloudject.m_PosA;
		m_PosB	= cloudject.m_PosB;
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
	PointCloudPtr m_ViewA, m_ViewB;
	PointT m_PosA, m_PosB;
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
	PointCloudPtr getViewA() { return CloudjectBase<PointT,U>::getViewA(); }
	PointCloudPtr getViewB() { return CloudjectBase<PointT,U>::getViewB(); }

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


	PointCloudPtr getViewA() { return CloudjectBase<PointT,pcl::FPFHSignature33>::getViewA(); }

	PointCloudPtr getViewB() { return CloudjectBase<PointT,pcl::FPFHSignature33>::getViewB(); }


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
