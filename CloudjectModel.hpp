#pragma once

#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>

#include "Cloudject.hpp"

#include <vector>

template<typename PointT, typename SignatureT>
class CloudjectModelBase
{

	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;

public:
	CloudjectModelBase() {}

	CloudjectModelBase(int ID, int nViewpoints = 3, float leafSize = 0.0)
		: m_ID(ID), m_NViews(nViewpoints), m_LeafSize(leafSize)
	{
	}

	~CloudjectModelBase(void) {}

	int getID() { return m_ID; }

	void addView(PointCloudPtr pView)
	{
		if (m_LeafSize > 0.0)
		{
			PointCloudPtr pViewF (new PointCloud);
			
			pcl::ApproximateVoxelGrid<PointT> avg;
			avg.setInputCloud(pView);
			avg.setLeafSize(m_LeafSize, m_LeafSize, m_LeafSize);
			avg.filter(*pViewF);

			pViewF.swap(pView);

			m_ViewClouds.push_back(pView);
		}
		else
		{
			PointCloudPtr pViewCpy (new PointCloud);
			pcl::copyPointCloud(*pView, *pViewCpy);

			m_ViewClouds.push_back(pView);
		}
	}

protected:

	std::vector<PointCloudPtr> m_ViewClouds;

	int m_NViews;
	float m_LeafSize; // in case of downsampling

private:
	
	int m_ID;
};


template<typename PointT, typename SignatureT>
class CloudjectModel : public CloudjectModelBase<PointT,SignatureT>
{

	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;

public:
	CloudjectModel(void) : CloudjectModelBase<PointT,SignatureT> {}

	CloudjectModel(int ID, int nViewpoints = 3, float leafSize = 0.0)
		: CloudjectModelBase<PointT,SignatureT>(ID, nViewpoints, leafSize) {}

	virtual ~CloudjectModel() {}

	int getID() { return CloudjectModelBase<PointT, SignatureT>::getID(); }

	void addView(PointCloudPtr pCloud)
	{
		CloudjectModelBase<PointT,SignatureT>::addView(pCloud);
	}
};


template<typename PointT>
class CloudjectModel<PointT, pcl::FPFHSignature33> : public CloudjectModelBase<PointT, pcl::FPFHSignature33>
{
	//typedef pcl::PointXYZ PointT;
	typedef pcl::FPFHSignature33 SignatureT;
	typedef pcl::PointCloud<SignatureT> Descriptor;
	typedef pcl::PointCloud<SignatureT>::Ptr DescriptorPtr;
	typedef Cloudject<PointT, SignatureT> Cloudject;
	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;

public:
	//CloudjectModel(void)
	//	: CloudjectModelBase<PointT, SignatureT>() 
	//{}

	CloudjectModel(int ID, int nViewpoints = 3, float leafSize = 0.0)
		: CloudjectModelBase<PointT, SignatureT>(ID, nViewpoints, leafSize) {}


	virtual ~CloudjectModel() {}


	int getID() { return CloudjectModelBase<PointT, SignatureT>::getID(); }
	

	void describe(float normalRadius, float fpfhRadius)
	{
		for (int i = 0; i < m_NViews; i++)
		{
			DescriptorPtr pDescriptor (new Descriptor);
			describeView(m_ViewClouds[i], normalRadius, fpfhRadius, *pDescriptor);
			m_ViewsDescriptors.push_back(pDescriptor);
		}
	}

	void describe(float leafSize, float normalRadius, float fpfhRadius)
	{
		for (int i = 0; i < m_NViews; i++)
		{
			DescriptorPtr pDescriptor (new Descriptor);
			describeView(m_ViewClouds[i], leafSize, normalRadius, fpfhRadius, *pDescriptor);
			m_ViewsDescriptors.push_back(pDescriptor);
		}
	}


	void describeView(PointCloudPtr pView, 
					  float leafSize, float normalRadius, float fpfhRadius,
					  Descriptor& descriptor)
	{
		PointCloudPtr pViewF (new PointCloud);

		pcl::ApproximateVoxelGrid<PointT> avg;
		avg.setInputCloud(pView);
		avg.setLeafSize(leafSize, leafSize, leafSize);
		avg.filter(*pViewF);

		pViewF.swap(pView);

		describeView(pView, normalRadius, fpfhRadius, descriptor);
	}


	void describeView(PointCloudPtr pView, 
					  float normalRadius, float fpfhRadius,
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


	float match(Cloudject c)
	{
		if (c.getType() == Cloudject::OneView)
			return matchView(c.getDescriptionA());
		else
			return ( matchView(c.getDescriptionA()) + matchView(c.getDescriptionB()) ) / 2.0;
	}


	float matchView(DescriptorPtr descriptor)
	{
		// Auxiliary structures: to not match against a model point more than once

		std::vector<int> numOfMatches;
		numOfMatches.resize(m_ViewsDescriptors.size(), 0);

		std::vector<std::vector<bool> > matches;

		matches.resize(m_ViewsDescriptors.size());
		for (int i = 0; i < matches.size(); i++)
			matches[i].resize(m_ViewsDescriptors[i]->points.size(), false);

		// Match

		float accDistToSig = 0;

		float minDistToP, dist; // inner-loop vars
		int minIdxV = -1, minIdxP = -1;
		int numOfTotalMatches = 0;

		for (int p = 0; p < descriptor->points.size(); p++)
		{
			bool freeCorrespondences = false; // there is any point to match against in the views of the model?

			minDistToP = std::numeric_limits<float>::infinity(); // min distance to other point histogram
		
			for (int i = 0; i < m_ViewsDescriptors.size() && numOfMatches[i] < m_ViewsDescriptors[i]->points.size(); i++) 
			{
				for (int j = 0; j < m_ViewsDescriptors[i]->points.size(); j++)
				{
					if ( freeCorrespondences = !(matches[i][j]) ) // A point in a view can only be matched one time against
					{
						dist = battacharyyaDistanceFPFHSignatures( descriptor->points[p], m_ViewsDescriptors[i]->points[j]/*, minDistToP*/);

						if (dist < minDistToP) // not matched yet and minimum
						{
							minDistToP = dist;
							minIdxV = i;
							minIdxP = j;
						}
					}
				}
			}
			
			// if it is not "true", minDist is infinity. Not to accumulate infinity :S
			// And dealing with another kinds of errors
			//if ( freeCorrespondences && !(minIdx < 0 || minIdxP < 0) )
			if (minDistToP < std::numeric_limits<float>::infinity())
			{
				accDistToSig += minDistToP;
				numOfTotalMatches ++;
				
				numOfMatches[minIdxV] ++; // aux var: easy way to know when all the points in a model have been matched
				matches[minIdxV][minIdxP] = true; // aux var: to know when a certain point in a certian model have already matched
			}
		}

		// Normalization: to deal with partial occlusions
		//float factor = (descriptor->points.size() / (float) averageNumOfPointsInModels());
		return accDistToSig / numOfTotalMatches;// / factor;
	}


	float battacharyyaDistanceFPFHSignatures(SignatureT s1, SignatureT s2)
	{
		float accSqProd = 0;
		float accS1 = 0;
		float accS2 = 0;
		for (int b = 0; b < 33; b++)
		{
			accSqProd += sqrt(s1.histogram[b] * s2.histogram[b]);
			accS1 += s1.histogram[b];
			accS2 += s2.histogram[b];
		}

		float f = 1.0 / sqrt((accS1/33) * (accS2/33) * (33*33));

		return sqrt(1 - f * accSqProd);
	}


	float euclideanDistanceFPFHSignatures(SignatureT s1, SignatureT s2)
	{
		float acc = 0;
		for (int b = 0; b < 33; b++)
		{
			acc += powf(s1.histogram[b] - s2.histogram[b], 2.0);
		}

		return sqrtf(acc);
	}


	float euclideanDistanceFPFHSignatures(SignatureT s1, SignatureT s2, float thresh)
	{
		float acc = 0;
		for (int b = 0; b < 33; b++)
		{
			if (sqrtf(acc) >= thresh)
				return thresh;

			acc += powf(s1.histogram[b] - s2.histogram[b], 2.0);
		}

		return sqrtf(acc);
	}


	float averageNumOfPointsInModels()
	{
		float acc = .0f;

		for (int i = 0; i < m_ViewsDescriptors.size(); i++)
			acc += m_ViewsDescriptors[i]->points.size();

		return acc / m_ViewsDescriptors.size();
	}

	void addView(PointCloudPtr pCloud)
	{
		CloudjectModelBase<PointT,SignatureT>::addView(pCloud);
	}

private:
	std::vector<DescriptorPtr> m_ViewsDescriptors;
};

