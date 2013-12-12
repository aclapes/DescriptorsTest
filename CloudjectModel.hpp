#pragma once

#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/pfhrgb.h>

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

			m_ViewClouds.push_back(pViewF);
		}
		else
		{
			PointCloudPtr pViewCpy (new PointCloud);
			pcl::copyPointCloud(*pView, *pViewCpy);

			m_ViewClouds.push_back(pViewCpy);
		}

		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*m_ViewClouds[m_ViewClouds.size()-1], centroid);

		PointT c;
		c.x = centroid.x();
		c.y = centroid.y();
		c.z = centroid.z();

		m_ViewCentroids.push_back(c);

		float medianDist = medianDistanceToCentroid(m_ViewClouds[m_ViewClouds.size()-1], c);

		m_MedianDistsToViewCentroids.push_back(medianDist);
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


	// Returns the average number of points among the views of the model
	float averageNumOfPointsInModels()
	{
		float acc = .0f;

		for (int i = 0; i < m_ViewsDescriptors.size(); i++)
			acc += m_ViewsDescriptors[i]->points.size();

		return acc / m_ViewsDescriptors.size();
	}


	float averageMedianDistanceToCentroids()
	{
		float acc = .0f;

		for (int i = 0; i < m_MedianDistsToViewCentroids.size(); i++)
			acc += m_MedianDistsToViewCentroids[i];

		return acc / m_MedianDistsToViewCentroids.size();
	}

protected:

	std::vector<PointCloudPtr> m_ViewClouds;
	std::vector<PointT> m_ViewCentroids;
	std::vector<float> m_MedianDistsToViewCentroids;

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

	float euclideanDistance() { return CloudjectModelBase<PointT,SignatureT>::euclideanDistance(); }

	float medianDistanceToCentroid(PointCloudPtr pCloud, PointT centroid)
	{
		return CloudjectModelBase<PointT,SignatureT>::medianDistanceToCentroid(pCloud, centroid);
	}

	float averageNumOfPointsInModels() { return CloudjectModelBase<PointT,SignatureT>::averageNumOfPointsInModels(); }

	float averageMedianDistanceToCentroids() { return CloudjectModelBase<PointT,SignatureT>::averageMedianDistanceToCentroids(); }
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

	CloudjectModel(int ID, int nViewpoints = 3, float leafSize = 0.0, float pointRejectionThresh = 1.0, float ratioRejectionThresh = 1.0,
		float sigmaPenaltyThresh = 0.1) : CloudjectModelBase<PointT, SignatureT>(ID, nViewpoints, leafSize), 
		m_PointRejectionThresh(pointRejectionThresh), m_RatioRejectionThresh(ratioRejectionThresh), m_SigmaPenaltyThresh(sigmaPenaltyThresh)
	{}


	virtual ~CloudjectModel() {}


	int getID() { return CloudjectModelBase<PointT, SignatureT>::getID(); }
	

	float euclideanDistance() { return CloudjectModelBase<PointT, pcl::FPFHSignature33>::euclideanDistance(); }


	float medianDistanceToCentroid(PointCloudPtr pCloud, PointT centroid)
	{ return CloudjectModelBase<PointT, pcl::FPFHSignature33>::medianDistanceToCentroid(pCloud, centroid); }
	

	float averageNumOfPointsInModels() 
	{ return CloudjectModelBase<PointT, pcl::FPFHSignature33>::averageNumOfPointsInModels(); }


	float averageMedianDistanceToCentroids() 
	{ return CloudjectModelBase<PointT, pcl::FPFHSignature33>::averageMedianDistanceToCentroids(); }


	// Describe all the model views
	void describe(float normalRadius, float fpfhRadius)
	{
		for (int i = 0; i < m_NViews; i++)
		{
			DescriptorPtr pDescriptor (new Descriptor);
			describeView(m_ViewClouds[i], normalRadius, fpfhRadius, *pDescriptor);
			m_ViewsDescriptors.push_back(pDescriptor);
		}
	}


	// Describe all the model views, performing
	// a prior downsampling to speed up the process
	void describe(float leafSize, float normalRadius, float fpfhRadius)
	{
		for (int i = 0; i < m_NViews; i++)
		{
			DescriptorPtr pDescriptor (new Descriptor);
			describeView(m_ViewClouds[i], leafSize, normalRadius, fpfhRadius, *pDescriptor);
			m_ViewsDescriptors.push_back(pDescriptor);
		}
	}


	// Compute the description of a view, performing
	// a prior downsampling to speed up the process
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


	// Compute the description of a view, actually
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

		pcl::FPFHEstimationOMP<PointT, pcl::Normal, SignatureT> fpfh;
		fpfh.setInputCloud (pView);
		fpfh.setInputNormals (pNormals);

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


	// Returns the score of matching a cloudject against the model
	float match(Cloudject c)
	{
		float sigma = m_SigmaPenaltyThresh;

		if (c.getType() == Cloudject::OneView)
		{
			float score = matchView(c.getDescriptionA());
			float diff = c.medianDistToCentroidInViewA() - averageMedianDistanceToCentroids();
			float norm = (1.0 / sigma * std::sqrtf(2 * 3.14159)) * std::expf(-0.5 * powf(diff / sigma, 2));
			return score * norm;
		}
		else
		{
			float scoreA = matchView(c.getDescriptionA());
			float scoreB = matchView(c.getDescriptionB());

			float diffA = c.medianDistToCentroidInViewA() - averageMedianDistanceToCentroids();
			float diffB = c.medianDistToCentroidInViewB() - averageMedianDistanceToCentroids();

			float normA = (1.0 / sigma * std::sqrtf(2 * 3.14159)) * std::expf(-0.5 * powf(diffA / sigma, 2));
			float normB = (1.0 / sigma * std::sqrtf(2 * 3.14159)) * std::expf(-0.5 * powf(diffB / sigma, 2));

			return ((scoreA * normA) + (scoreB * normB)) / 2.0;
		}

		//if (c.getType() == Cloudject::OneView)
		//	return matchView(c.getDescriptionA());
		//else
		//	return (matchView(c.getDescriptionA()) + matchView(c.getDescriptionB())) / 2.0;
	}


	// Returns the score of matching a description of a certain cloudject's view against the model views' descriptions
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

		float minDistToP, ndMinDist, dist; // inner-loop vars
		int minIdxV = -1, minIdxP = -1;
		int numOfTotalMatches = 0;

		for (int p = 0; p < descriptor->points.size(); p++)
		{
			bool freeCorrespondences = false; // there is any point to match against in the views of the model?

			minDistToP = std::numeric_limits<float>::infinity(); // min distance to other point histogram
			ndMinDist =  std::numeric_limits<float>::infinity();
		
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
						else if (dist < ndMinDist)
						{
							ndMinDist = dist;
						}
					}
				}
			}
			
			// if it is not "true", minDist is infinity. Not to accumulate infinity :S
			// And dealing with another kinds of errors
			//if ( fereCorrespondences && !(minIdx < 0 || minIdxP < 0) )
			if (minDistToP <= m_PointRejectionThresh/* && (minDistToP/ndMinDist) < m_RatioRejectionThresh*/)
			{
				accDistToSig += minDistToP;
				numOfTotalMatches ++;
				
				numOfMatches[minIdxV] ++; // aux var: easy way to know when all the points in a model have been matched
				matches[minIdxV][minIdxP] = true; // aux var: to know when a certain point in a certian model have already matched
			}
		}

		// Normalization: to deal with partial occlusions
		//float factor = (descriptor->points.size() / (float) averageNumOfPointsInModels());
		
		float avgDist = accDistToSig / numOfTotalMatches;
		float score =  1 - avgDist;

		return score; // / descriptor->points.size());
	}


	// Returns the battacharyya distance between two fpfh signatures, which are actually histograms.
	// This is a normalized [0,1] distance
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


	// Returns the euclidean distance between two fpfh signatures, which are actually histograms
	float euclideanDistanceFPFHSignatures(SignatureT s1, SignatureT s2)
	{
		float acc = 0;
		for (int b = 0; b < 33; b++)
		{
			acc += powf(s1.histogram[b] - s2.histogram[b], 2.0);
		}

		return sqrtf(acc);
	}


	// Returns the euclidean distance between two fpfh signatures, which are actually histograms
	// subtracting bin-by-bin while the square root of the accumulated subtractions are lower than
	// a threshold. Otherwise, return the threshold.
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

	
	// Add a view (point cloud) to the model, up to the desired number of views
	void addView(PointCloudPtr pCloud)
	{
		CloudjectModelBase<PointT,SignatureT>::addView(pCloud);
	}

private:

	//
	// Private members
	// 

	// The descriptions of the different views
	std::vector<DescriptorPtr>		m_ViewsDescriptors;
	// A valid best correspondence should be a distance below it (experimentally selected)
	float							m_PointRejectionThresh;
	float							m_RatioRejectionThresh;
	float							m_SigmaPenaltyThresh;
};

//
//template<typename PointT>
//class CloudjectModel<PointT, pcl::PFHRGBSignature250> : public CloudjectModelBase<PointT, pcl::PFHRGBSignature250>
//{
//	//typedef pcl::PointXYZ PointT;
//	typedef pcl::PFHRGBSignature250 SignatureT;
//	typedef pcl::PointCloud<SignatureT> Descriptor;
//	typedef pcl::PointCloud<SignatureT>::Ptr DescriptorPtr;
//	typedef Cloudject<PointT, SignatureT> Cloudject;
//	typedef typename pcl::PointCloud<PointT> PointCloud;
//	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
//
//public:
//
//	CloudjectModel(int ID, int nViewpoints = 3, float leafSize = 0.0, float pointRejectionThresh = 0.5)
//		: CloudjectModelBase<PointT, SignatureT>(ID, nViewpoints, leafSize), m_PointRejectionThresh(pointRejectionThresh) {}
//
//
//	virtual ~CloudjectModel() {}
//
//
//	int getID() { return CloudjectModelBase<PointT, SignatureT>::getID(); }
//	
//
//	// Describe all the model views
//	void describe(float normalRadius, float featureRadius)
//	{
//		for (int i = 0; i < m_NViews; i++)
//		{
//			DescriptorPtr pDescriptor (new Descriptor);
//			describeView(m_ViewClouds[i], normalRadius, featureRadius, *pDescriptor);
//			m_ViewsDescriptors.push_back(pDescriptor);
//		}
//	}
//
//
//	// Describe all the model views, performing
//	// a prior downsampling to speed up the process
//	void describe(float leafSize, float normalRadius, float featureRadius)
//	{
//		for (int i = 0; i < m_NViews; i++)
//		{
//			DescriptorPtr pDescriptor (new Descriptor);
//			describeView(m_ViewClouds[i], leafSize, normalRadius, featureRadius, *pDescriptor);
//			m_ViewsDescriptors.push_back(pDescriptor);
//		}
//	}
//
//
//	// Compute the description of a view, performing
//	// a prior downsampling to speed up the process
//	void describeView(PointCloudPtr pView, 
//					  float leafSize, float normalRadius, float featureRadius,
//					  Descriptor& descriptor)
//	{
//		PointCloudPtr pViewF (new PointCloud);
//
//		pcl::ApproximateVoxelGrid<PointT> avg;
//		avg.setInputCloud(pView);
//		avg.setLeafSize(leafSize, leafSize, leafSize);
//		avg.filter(*pViewF);
//
//		pViewF.swap(pView);
//
//		describeView(pView, normalRadius, featureRadius, descriptor);
//	}
//
//
//	// Compute the description of a view, actually
//	void describeView(PointCloudPtr pView, 
//					  float normalRadius, float featureRadius,
//					  Descriptor& descriptor)
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
//		// PFHRGB description extraction
//		//
//
//		pcl::PFHRGBEstimation<PointT, pcl::Normal, SignatureT> pfhrgb;
//		pfhrgb.setInputCloud (pView);
//		pfhrgb.setInputNormals (pNormals);
//		// alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);
//
//		// Create an empty kdtree representation, and pass it to the FPFH estimation object.
//		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
//		tree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>);
//		pfhrgb.setSearchMethod (tree);
//		
//		// Output datasets
//		//pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr pfhs (new pcl::PointCloud<pcl::PFHRGBSignature250> ());
//
//		// Use all neighbors in a sphere of radius 5cm
//		// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
//		pfhrgb.setRadiusSearch (featureRadius);
//
//		// Compute the features
//		pfhrgb.compute (descriptor);
//	}
//
//
//	// Returns the score of matching a cloudject against the model
//	float match(Cloudject c)
//	{
//		if (c.getType() == Cloudject::OneView)
//			return matchView(c.getDescriptionA());
//		else
//			return ( matchView(c.getDescriptionA()) + matchView(c.getDescriptionB()) ) / 2.0;
//	}
//
//
//	// Returns the score of matching a description of a certain cloudject's view against the model views' descriptions
//	float matchView(DescriptorPtr descriptor)
//	{
//		// Auxiliary structures: to not match against a model point more than once
//
//		std::vector<int> numOfMatches;
//		numOfMatches.resize(m_ViewsDescriptors.size(), 0);
//
//		std::vector<std::vector<bool> > matches;
//
//		matches.resize(m_ViewsDescriptors.size());
//		for (int i = 0; i < matches.size(); i++)
//			matches[i].resize(m_ViewsDescriptors[i]->points.size(), false);
//
//		// Match
//
//		float accDistToSig = 0;
//
//		float minDistToP, dist; // inner-loop vars
//		int minIdxV = -1, minIdxP = -1;
//		int numOfTotalMatches = 0;
//
//		for (int p = 0; p < descriptor->points.size(); p++)
//		{
//			bool freeCorrespondences = false; // there is any point to match against in the views of the model?
//
//			minDistToP = std::numeric_limits<float>::infinity(); // min distance to other point histogram
//		
//			for (int i = 0; i < m_ViewsDescriptors.size() && numOfMatches[i] < m_ViewsDescriptors[i]->points.size(); i++) 
//			{
//				for (int j = 0; j < m_ViewsDescriptors[i]->points.size(); j++)
//				{
//					if ( freeCorrespondences = !(matches[i][j]) ) // A point in a view can only be matched one time against
//					{
//						dist = battacharyyaDistanceFPFHSignatures( descriptor->points[p], m_ViewsDescriptors[i]->points[j]/*, minDistToP*/);
//
//						if (dist < minDistToP) // not matched yet and minimum
//						{
//							minDistToP = dist;
//							minIdxV = i;
//							minIdxP = j;
//						}
//					}
//				}
//			}
//			
//			// if it is not "true", minDist is infinity. Not to accumulate infinity :S
//			// And dealing with another kinds of errors
//			//if ( freeCorrespondences && !(minIdx < 0 || minIdxP < 0) )
//			if (/*minDistToP < std::numeric_limits<float>::infinity() &&*/minDistToP <= m_PointRejectionThresh)
//			{
//				accDistToSig += minDistToP;
//				numOfTotalMatches ++;
//				
//				numOfMatches[minIdxV] ++; // aux var: easy way to know when all the points in a model have been matched
//				matches[minIdxV][minIdxP] = true; // aux var: to know when a certain point in a certian model have already matched
//			}
//		}
//
//		// Normalization: to deal with partial occlusions
//		//float factor = (descriptor->points.size() / (float) averageNumOfPointsInModels());
//		
//		float avgDist = accDistToSig / numOfTotalMatches;
//		float score =  1 - avgDist;
//
//		return score * (numOfTotalMatches / descriptor->points.size());
//	}
//
//
//	// Returns the battacharyya distance between two fpfh signatures, which are actually histograms.
//	// This is a normalized [0,1] distance
//	float battacharyyaDistanceFPFHSignatures(SignatureT s1, SignatureT s2)
//	{
//		float accSqProd = 0;
//		float accS1 = 0;
//		float accS2 = 0;
//		for (int b = 0; b < 250; b++)
//		{
//			accSqProd += sqrt(s1.histogram[b] * s2.histogram[b]);
//			accS1 += s1.histogram[b];
//			accS2 += s2.histogram[b];
//		}
//
//		float f = 1.0 / sqrt((accS1/250) * (accS2/250) * (250*250));
//
//		return sqrt(1 - f * accSqProd);
//	}
//
//
//	// Returns the euclidean distance between two fpfh signatures, which are actually histograms
//	float euclideanDistanceFPFHSignatures(SignatureT s1, SignatureT s2)
//	{
//		float acc = 0;
//		for (int b = 0; b < 250; b++)
//		{
//			acc += powf(s1.histogram[b] - s2.histogram[b], 2.0);
//		}
//
//		return sqrtf(acc);
//	}
//
//
//	// Returns the euclidean distance between two fpfh signatures, which are actually histograms
//	// subtracting bin-by-bin while the square root of the accumulated subtractions are lower than
//	// a threshold. Otherwise, return the threshold.
//	float euclideanDistanceFPFHSignatures(SignatureT s1, SignatureT s2, float thresh)
//	{
//		float acc = 0;
//		for (int b = 0; b < 250; b++)
//		{
//			if (sqrtf(acc) >= thresh)
//				return thresh;
//
//			acc += powf(s1.histogram[b] - s2.histogram[b], 2.0);
//		}
//
//		return sqrtf(acc);
//	}
//
//	
//	// Add a view (point cloud) to the model, up to the desired number of views
//	void addView(PointCloudPtr pCloud)
//	{
//		CloudjectModelBase<PointT,SignatureT>::addView(pCloud);
//	}
//
//private:
//
//	//
//	// Private members
//	// 
//
//	// The descriptions of the different views
//	std::vector<DescriptorPtr>		m_ViewsDescriptors;
//	// A valid best correspondence should be a distance below it (experimentally selected)
//	float							m_PointRejectionThresh;
//};
//
