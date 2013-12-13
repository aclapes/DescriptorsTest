#pragma once

#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
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

		for (int i = 0; i < m_ViewClouds.size(); i++)
			acc += m_ViewClouds[i]->points.size();

		return acc / m_ViewClouds.size();
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
class LFCloudjectModelBase : public CloudjectModelBase<PointT,SignatureT>
{
	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
	typedef typename pcl::PointCloud<SignatureT> Descriptor;
	typedef typename pcl::PointCloud<SignatureT>::Ptr DescriptorPtr;

	//typedef typename LFCloudject<PointT,SignatureT> LFCloudject;

protected:
	LFCloudjectModelBase(void) 
		: CloudjectModelBase<PointT,SignatureT> {}

	LFCloudjectModelBase(int ID, int nViewpoints = 3, float leafSize = 0.0, float pointRejectionThresh = 1.0, 
		float ratioRejectionThresh = 1.0, int sizePenalty = 1, float sigmaPenaltyThresh = 0.1)
		: CloudjectModelBase<PointT,SignatureT>(ID, nViewpoints, leafSize), 
		  m_PointRejectionThresh(pointRejectionThresh), m_RatioRejectionThresh(ratioRejectionThresh), 
		  m_SizePenalty(sizePenalty), m_SigmaPenaltyThresh(sigmaPenaltyThresh) 
	{}

	virtual ~LFCloudjectModelBase() {}

	int getID() { return CloudjectModelBase<PointT, SignatureT>::getID(); }

	void addView(PointCloudPtr pCloud) { CloudjectModelBase<PointT,SignatureT>::addView(pCloud); }

	float euclideanDistance() { return CloudjectModelBase<PointT,SignatureT>::euclideanDistance(); }
	float medianDistanceToCentroid(PointCloudPtr pCloud, PointT centroid)
	{ return CloudjectModelBase<PointT,SignatureT>::medianDistanceToCentroid(pCloud, centroid); }

	float averageNumOfPointsInModels() { return CloudjectModelBase<PointT,SignatureT>::averageNumOfPointsInModels(); }
	float averageMedianDistanceToCentroids() { return CloudjectModelBase<PointT,SignatureT>::averageMedianDistanceToCentroids(); }


	// Returns the score of matching a cloudject against the model
	float match(LFCloudject<PointT,SignatureT> c)
	{
		float sigma = m_SigmaPenaltyThresh;

		if (c.getType() == LFCloudject<PointT,SignatureT>::OneView)
		{
			float score = matchView(c.getDescriptionA());
			float penalty;
			
			if (getSizePenalty() == SizePenalty::NumOfPoints)
			{
				float ratio = ((float) c.getNumOfPointsInViewA()) / averageNumOfPointsInModels();
				
				float x;
				ratio <= 1 ? (x = ratio) : ( x = 1 + (1 - (1 / ratio)) );

				penalty = (1.0 / sigma * std::sqrtf(2 * 3.14159)) * std::expf(-0.5 * powf((x-1)/sigma, 2));
			}
			else // SizePenalty::MeanDistToCentroid
			{
				float diff = c.medianDistToCentroidInViewA() - averageMedianDistanceToCentroids();
				penalty = (1.0 / sigma * std::sqrtf(2 * 3.14159)) * std::expf(-0.5 * powf(diff/sigma, 2));
			}

			return score * penalty;
		}
		else
		{
			float scoreA = matchView(c.getDescriptionA());
			float scoreB = matchView(c.getDescriptionB());

			float penaltyA, penaltyB;

			if (getSizePenalty() == SizePenalty::NumOfPoints)
			{
				float ratioA = ((float) c.getNumOfPointsInViewA()) / averageNumOfPointsInModels();
				float ratioB = ((float) c.getNumOfPointsInViewB()) / averageNumOfPointsInModels();
				float xA, xB;
				
				ratioA <= 1 ? (xA = ratioA) : ( xA = 1 + (1 - (1 / ratioA)) );
				ratioB <= 1 ? (xB = ratioB) : ( xB = 1 + (1 - (1 / ratioB)) );

				penaltyA = (1.0 / sigma * std::sqrtf(2 * 3.14159)) * std::expf(-0.5 * powf((xA-1)/sigma, 2));
				penaltyB = (1.0 / sigma * std::sqrtf(2 * 3.14159)) * std::expf(-0.5 * powf((xB-1)/sigma, 2));
			}
			else // SizePenalty::MeanDistToCentroid
			{
				float diffA = c.medianDistToCentroidInViewA() - averageMedianDistanceToCentroids();
				float diffB = c.medianDistToCentroidInViewB() - averageMedianDistanceToCentroids();

				penaltyA = (1.0 / sigma * std::sqrtf(2 * 3.14159)) * std::expf(-0.5 * powf(diffA/sigma, 2));
				penaltyB = (1.0 / sigma * std::sqrtf(2 * 3.14159)) * std::expf(-0.5 * powf(diffB/sigma, 2));
			}

			// TODO: more sophisticated fusion
			return ((scoreA * penaltyA) + (scoreB * penaltyB)) / 2.0;
		}
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
						dist = battacharyyaDistanceSignatures( descriptor->points[p], m_ViewsDescriptors[i]->points[j]/*, minDistToP*/);

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
	float battacharyyaDistanceSignatures(SignatureT s1, SignatureT s2)
	{
		float accSqProd = 0;
		float accS1 = 0;
		float accS2 = 0;
		for (int b = 0; b < sizeof(s1.histogram) / sizeof(s1.histogram[0]); b++)
		{
			accSqProd += sqrt(s1.histogram[b] * s2.histogram[b]);
			accS1 += s1.histogram[b];
			accS2 += s2.histogram[b];
		}

		float f = 1.0 / sqrt((accS1/33) * (accS2/33) * (33*33));

		return sqrt(1 - f * accSqProd);
	}


	// Returns the euclidean distance between two fpfh signatures, which are actually histograms
	float euclideanDistanceSignatures(SignatureT s1, SignatureT s2)
	{
		float acc = 0;
		for (int b = 0; b < sizeof(s1.histogram) / sizeof(s1.histogram[0]); b++)
		{
			acc += powf(s1.histogram[b] - s2.histogram[b], 2.0);
		}

		return sqrtf(acc);
	}


	// Returns the euclidean distance between two fpfh signatures, which are actually histograms
	// subtracting bin-by-bin while the square root of the accumulated subtractions are lower than
	// a threshold. Otherwise, return the threshold.
	float euclideanDistanceSignatures(SignatureT s1, SignatureT s2, float thresh)
	{
		float acc = 0;
		for (int b = 0; b < sizeof(s1.histogram) / sizeof(s1.histogram[0]); b++)
		{
			if (sqrtf(acc) >= thresh)
				return thresh;

			acc += powf(s1.histogram[b] - s2.histogram[b], 2.0);
		}

		return sqrtf(acc);
	}


	int getSizePenalty()
	{
		return m_SizePenalty;
	}


	//
	// Protected members
	// 

	// The descriptions of the different views
	std::vector<DescriptorPtr>		m_ViewsDescriptors;
	// A valid best correspondence should be a distance below it (experimentally selected)
	float							m_PointRejectionThresh;
	float							m_RatioRejectionThresh;
	float							m_SigmaPenaltyThresh;

	int								m_SizePenalty;
	enum							SizePenalty { NumOfPoints, MedianDistToCentroid };

private:

	float penalty(float diff)
	{

	}
};


//
// Templates
//

// Generic template
template<typename PointT, typename SignatureT>
class LFCloudjectModel : public LFCloudjectModelBase<PointT,SignatureT>
{};

// Partially specialized template
template<typename PointT>
class LFCloudjectModel<PointT, pcl::FPFHSignature33> : public LFCloudjectModelBase<PointT, pcl::FPFHSignature33>
{
	typedef pcl::PointCloud<pcl::FPFHSignature33> Descriptor;
	typedef pcl::PointCloud<pcl::FPFHSignature33>::Ptr DescriptorPtr;
	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;

	typedef typename LFCloudject<PointT,pcl::FPFHSignature33> LFCloudject;

public:

	LFCloudjectModel(int ID, int nViewpoints = 3, float leafSize = 0.0, float pointRejectionThresh = 1.0, 
		float ratioRejectionThresh = 1.0, int sizePenalty = 1, float sigmaPenaltyThresh = 0.1)
		: LFCloudjectModelBase<PointT,pcl::FPFHSignature33>(ID, nViewpoints, leafSize,
		  pointRejectionThresh, ratioRejectionThresh, sizePenalty, sigmaPenaltyThresh)
	{}

	virtual ~LFCloudjectModel() {}

	int getID() { return LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::getID(); }
	
	void addView(PointCloudPtr pCloud) { LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::addView(pCloud); }

	float euclideanDistance() { return LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::euclideanDistance(); }
	float medianDistanceToCentroid(PointCloudPtr pCloud, PointT centroid)
	{ return LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::medianDistanceToCentroid(pCloud, centroid); }
	
	float averageNumOfPointsInModels() 
	{ return LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::averageNumOfPointsInModels(); }
	float averageMedianDistanceToCentroids() 
	{ return LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::averageMedianDistanceToCentroids(); }

	float match(LFCloudject c)
	{ return LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::match(c); }

	int getSizePenalty()
	{ return LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::getSizePenalty(); }


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

private:
	float matchView(DescriptorPtr pDescriptor)
	{ return LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::matchView(pDescriptor); }

	float battacharyyaDistanceSignatures(pcl::FPFHSignature33 s1, pcl::FPFHSignature33 s2)
	{ return LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::battacharyyaDistanceSignatures(s1, s2); }

	float euclideanDistanceSignatures(pcl::FPFHSignature33 s1, pcl::FPFHSignature33 s2)
	{ return LFCloudjectModelBase<PointT,pcl::FPFHSignature3>::euclideanDistanceSignatures(s1, s2); }

	float euclideanDistanceSignatures(pcl::FPFHSignature33 s1, pcl::FPFHSignature33 s2, float thresh)
	{ return LFCloudjectModelBase<PointT,pcl::FPFHSignature3>::euclideanDistanceSignatures(s1, s2, thresh); }


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

		pcl::FPFHEstimation<PointT,pcl::Normal,pcl::FPFHSignature33> fpfh;
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
};


// Partially specialized template
template<typename PointT>
class LFCloudjectModel<PointT,pcl::PFHRGBSignature250> : public LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>
{
	typedef pcl::PointCloud<pcl::PFHRGBSignature250> Descriptor;
	typedef pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr DescriptorPtr;
	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;

	typedef typename LFCloudject<PointT,pcl::PFHRGBSignature250> LFCloudject;

public:

	LFCloudjectModel(int ID, int nViewpoints = 3, float leafSize = 0.0, float pointRejectionThresh = 1.0, 
		float ratioRejectionThresh = 1.0, int sizePenalty = 1, float sigmaPenaltyThresh = 0.1)
		: LFCloudjectModelBase<PointT, pcl::PFHRGBSignature250>(ID, nViewpoints, leafSize,
		  pointRejectionThresh, ratioRejectionThresh, sizePenalty, sigmaPenaltyThresh)
	{}

	virtual ~LFCloudjectModel() {}

	int getID() { return LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>::getID(); }
	
	void addView(PointCloudPtr pCloud) { LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>::addView(pCloud); }

	float euclideanDistance() { return LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>::euclideanDistance(); }
	float medianDistanceToCentroid(PointCloudPtr pCloud, PointT centroid)
	{ return LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>::medianDistanceToCentroid(pCloud, centroid); }
	
	float averageNumOfPointsInModels() 
	{ return LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>::averageNumOfPointsInModels(); }
	float averageMedianDistanceToCentroids() 
	{ return LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>::averageMedianDistanceToCentroids(); }

	float match(LFCloudject c)
	{ return LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>::match(c); }

	int getSizePenalty()
	{ return LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>::getSizePenalty(); }


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

private:
	float matchView(DescriptorPtr pDescriptor)
	{ return LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>::matchView(pDescriptor); }

	float battacharyyaDistanceSignatures(pcl::PFHRGBSignature250 s1, pcl::PFHRGBSignature250 s2)
	{ return LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>::battacharyyaDistanceSignatures(s1, s2); }

	float euclideanDistanceSignatures(pcl::PFHRGBSignature250 s1, pcl::PFHRGBSignature250 s2)
	{ return LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>::euclideanDistanceSignatures(s1, s2); }

	float euclideanDistanceSignatures(pcl::PFHRGBSignature250 s1, pcl::PFHRGBSignature250 s2, float thresh)
	{ return LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>::euclideanDistanceSignatures(s1, s2, thresh); }


	// Compute the description of a view, performing
	// a prior downsampling to speed up the process
	void describeView(PointCloudPtr pView, 
					  float leafSize, float normalRadius, float pfhrgbRadius,
					  Descriptor& descriptor)
	{
		PointCloudPtr pViewF (new PointCloud);

		pcl::ApproximateVoxelGrid<PointT> avg;
		avg.setInputCloud(pView);
		avg.setLeafSize(leafSize, leafSize, leafSize);
		avg.filter(*pViewF);

		pViewF.swap(pView);

		describeView(pView, normalRadius, pfhrgbRadius, descriptor);
	}


	// Compute the description of a view, actually
	void describeView(PointCloudPtr pView, 
					  float normalRadius, float pfhrgbRadius,
					  Descriptor& descriptor)
	{
		//
		// Normals preprocess
		//

		// Create the normal estimation class, and pass the input dataset to it
		pcl::NormalEstimation<PointT,pcl::Normal> ne;
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

		pcl::PFHRGBEstimation<PointT,pcl::Normal,pcl::PFHRGBSignature250> pfhrgb;
		pfhrgb.setInputCloud (pView);
		pfhrgb.setInputNormals (pNormals);

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
};