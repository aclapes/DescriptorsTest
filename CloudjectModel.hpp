#pragma once

#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfhrgb.h>

#include "Cloudject.hpp"

#include <vector>

using namespace std;

template<typename PointT, typename SignatureT>
class CloudjectModelBase
{
    
	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
    
public:
	CloudjectModelBase(float leafSize)
    : m_ID(0), m_Name(""), m_LeafSize(leafSize)
    {
    }
    
	CloudjectModelBase(int ID, string name, float leafSize = 0.0)
    : m_ID(ID), m_Name(name), m_LeafSize(leafSize)
	{
	}
    
    CloudjectModelBase(int ID, string name, vector<PointCloudPtr> views, float leafSize = 0.0)
    : m_ID(ID), m_Name(name), m_LeafSize(leafSize)
	{
        for (int v = 0; v < views.size(); v++)
        {
            addView(views[v]);
        }
	}
    
    CloudjectModelBase(int ID, string name, vector<string> viewsFilepaths, float leafSize = 0.0)
    : m_ID(ID), m_Name(name), m_LeafSize(leafSize)
	{
        for (int v = 0; v < viewsFilepaths.size(); v++)
        {
            PointCloudPtr pView (new PointCloud);
            
			pcl::PCDReader reader;
			reader.read(viewsFilepaths[v], *pView);
            
            addView(pView);
        }
	}
    
    CloudjectModelBase(const CloudjectModelBase& rhs)
	{
        *this = rhs;
	}
    
    CloudjectModelBase& operator=(const CloudjectModelBase& rhs)
	{
        if (this != &rhs)
        {
            m_ID = rhs.m_ID;
            m_Name = rhs.m_Name;
            m_LeafSize = rhs.m_LeafSize;
            
            m_OriginalViews = rhs.m_OriginalViews;
            m_Views = rhs.m_Views;
            m_ViewCentroids = rhs.m_ViewCentroids;
            m_MedianDistsToCentroids = rhs.m_MedianDistsToCentroids;
        }
        
        return *this;
	}
    
    void addView(PointCloudPtr pView)
	{
        // Add the view itself
        
		// Get a copy of the view untouched
		PointCloudPtr pOriginalView (new PointCloud);
		*pOriginalView = *pView;
        
		// Get the downsampled version of the view
        if (m_LeafSize > 0.f)
			downsampleView(pOriginalView, m_LeafSize, *pView);
        
        // Get the precomputed view centroid
		pcl::PointXYZ centroidPt;
		Eigen::Vector4f centroidVc;
		pcl::compute3DCentroid(*pView, centroidVc);
        centroidPt.getVector4fMap() = centroidVc;
        
		// Get the precomputed view's median distance to centroid
		float medianDist = medianDistanceToCentroid(pView, centroidPt);
        
        // And... keep it properly
		m_OriginalViews.push_back(pOriginalView);
		m_Views.push_back(pView);
        m_MedianDistsToCentroids.push_back(medianDist);
	}
    
	int getID() { return m_ID; }
    void setID(int id) { m_ID = id; }
    
    string getName() { return m_Name; }
    void setName(string name) { m_Name = name; }
    
    float getDownsamplingSize() { return m_LeafSize; }
    
    int getNumOfViews() { return m_Views.size(); }
    
    PointCloudPtr getView(int v) { return m_Views[v]; }
    
    template<typename PointT1, typename PointT2>
	float euclideanDistance(PointT1 p1, PointT2 p2)
	{
		return sqrt(powf(p1.x - p2.x, 2) + powf(p1.y - p2.y, 2) + powf(p1.z - p2.z, 2));
	}
    
	float medianDistanceToCentroid(PointCloudPtr pCloud, pcl::PointXYZ centroid)
	{
		vector<float> distances;
        
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
        
        int medianIdx = distances.size() / 2;
		return distances[medianIdx];
	}
    
    
	// Returns the average number of points among the views of the model
	float averageNumOfPointsInModels()
	{
		float acc = 0.f;
        
		for (int v = 0; v < m_Views.size(); v++)
			acc += m_Views[v]->points.size();
        
		return acc / m_Views.size();
	}
    
	float averageMedianDistanceToCentroids()
	{
		float acc = 0.f;
        
		for (int v = 0; v < m_MedianDistsToCentroids.size(); v++)
			acc += m_MedianDistsToCentroids[v];
        
		return acc / m_MedianDistsToCentroids.size();
	}
    
    void downsample(float leafSize)
	{
		for (int v = 0; v < m_Views.size(); v++)
		{
			PointCloudPtr pDwView (new PointCloud);
            
			if (leafSize > m_LeafSize)
				downsampleView( (m_Views[v]->empty() ? m_OriginalViews[v] : m_Views[v]), leafSize, *pDwView );
			else if (leafSize < m_LeafSize)
				downsampleView( m_OriginalViews[v], leafSize, *pDwView );
            
			m_Views[v] = pDwView;
		}
        
        m_LeafSize = leafSize;
	}
    
protected:
    
    void downsampleView(PointCloudPtr pCloud, float leafSize, PointCloud& dwCloud)
	{
        if (leafSize == 0.f)
        {
            dwCloud = *pCloud;
        }
        else
        {
            pcl::VoxelGrid<PointT> avg;
            avg.setInputCloud(pCloud);
            avg.setLeafSize(leafSize, leafSize, leafSize);
            avg.filter(dwCloud);
        }
	}
    
    vector<PointCloudPtr> m_OriginalViews;
	vector<PointCloudPtr> m_Views;
	vector<pcl::PointXYZ> m_ViewCentroids;
	vector<float> m_MedianDistsToCentroids;
    
	float m_LeafSize; // in case of downsampling
    
private:
	
	int m_ID;
    string m_Name;
};


template<typename PointT, typename SignatureT>
class LFCloudjectModelBase : public CloudjectModelBase<PointT,SignatureT>
{
	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
	typedef typename pcl::PointCloud<SignatureT> Description;
	typedef typename pcl::PointCloud<SignatureT>::Ptr DescriptionPtr;
    
public:
	LFCloudjectModelBase(float leafSize = 0.f)
    : CloudjectModelBase<PointT,SignatureT>(leafSize) {}
    
	LFCloudjectModelBase(int ID, string name, float leafSize = 0.0, int penalty = 2, float pointRejectionThresh = 1.0, float ratioRejectionThresh = 1.0, float sigmaPenaltyThresh = 0.1)
    : CloudjectModelBase<PointT,SignatureT>(ID, name, leafSize), m_PointRejectionThresh(pointRejectionThresh), m_RatioRejectionThresh(ratioRejectionThresh), m_Penalty(penalty), m_SigmaPenaltyThresh(sigmaPenaltyThresh)
	{}
    
    LFCloudjectModelBase(int ID, string name, vector<PointCloudPtr> views, float leafSize = 0.0, int penalty = 2, float pointRejectionThresh = 1.0, float ratioRejectionThresh = 1.0, float sigmaPenaltyThresh = 0.1)
    : CloudjectModelBase<PointT,SignatureT>(ID, name, views, leafSize), m_PointRejectionThresh(pointRejectionThresh), m_RatioRejectionThresh(ratioRejectionThresh), m_Penalty(penalty), m_SigmaPenaltyThresh(sigmaPenaltyThresh)
	{}
    
    LFCloudjectModelBase(const LFCloudjectModelBase& rhs)
    : CloudjectModelBase<PointT,SignatureT>(rhs)
    {
        *this = rhs;
    }
    
    LFCloudjectModelBase& operator=(const LFCloudjectModelBase& rhs)
    {
        if (this != &rhs)
        {
            m_ViewsDescriptions = rhs.m_ViewsDescriptions;
            
            m_PointRejectionThresh = rhs.m_PointRejectionThresh;
            m_RatioRejectionThresh = rhs.m_RatioRejectionThresh;
            m_SigmaPenaltyThresh = rhs.m_SigmaPenaltyThresh;
            
            m_Penalty = rhs.m_Penalty;
        }
        
        return *this;
    }
    
    void setPointScoreRejectionThreshold(float pointRejectionThresh)
    {
        m_PointRejectionThresh = pointRejectionThresh;
    }
    
    void setPointRatioRejectionThreshold(float ratioRejectionThresh)
    {
        m_RatioRejectionThresh = ratioRejectionThresh;
    }
    
    void setSizePenaltyMode(int penalty)
    {
        m_Penalty = penalty;
    }
    
    void setSigmaPenaltyThreshold(float sigmaPenaltyThresh)
    {
        m_SigmaPenaltyThresh = sigmaPenaltyThresh;
    }
    
    //	int getID() { return CloudjectModelBase<PointT, SignatureT>::getID(); }
    //    string getName() { return CloudjectModelBase<PointT, SignatureT>::getName(); }
    //
    //    int getNumOfViews() { return CloudjectModelBase<PointT, SignatureT>::getNumOfViews(); }
    //
    //	void addView(PointCloudPtr pCloud) { CloudjectModelBase<PointT,SignatureT>::addView(pCloud); }
    //    PointCloudPtr getView(int i) { return CloudjectModelBase<PointT,SignatureT>::getView(i); }
    //
    //
    //	float euclideanDistance(PointT p1, PointT p2) { return CloudjectModelBase<PointT,SignatureT>::euclideanDistance(p1,p2); }
    //	float medianDistanceToCentroid(PointCloudPtr pCloud, PointT centroid)
    //	{ return CloudjectModelBase<PointT,SignatureT>::medianDistanceToCentroid(pCloud, centroid); }
    //
    //	float averageNumOfPointsInModels() { return CloudjectModelBase<PointT,SignatureT>::averageNumOfPointsInModels(); }
    //	float averageMedianDistanceToCentroids() { return CloudjectModelBase<PointT,SignatureT>::averageMedianDistanceToCentroids(); }
    
    void addViewDescription(DescriptionPtr pDescription)
    { m_ViewsDescriptions.push_back(pDescription); }
    DescriptionPtr getViewDescription(int i) { return m_ViewsDescriptions[i]; }
    
    // Test against this model. Returns distances (in each view) from points to matched model points (wo/ re-match)
    void getMinimumDistances(typename LFCloudject<PointT,SignatureT>::Ptr pCloudject, vector<vector<float> >& distances)
    {
        distances.clear();
        
        distances.resize(pCloudject->getNumOfViews());
        for (int v = 0; v < pCloudject->getNumOfViews(); v++)
        {
            vector<float> viewDistances;
            getMinimumDistancesToDescription(pCloudject->getDescription(v), viewDistances);
            
            distances[v] = viewDistances;
        }
    }
    
    static float distToScore(float dist)
    {
        return 1 - dist;
    }
    
    void getScores(typename LFCloudject<PointT,SignatureT>::Ptr pCloudject, vector<float>& scores)
    {
        scores.clear();
        
        vector<vector<float> > distances; // #{views} x #{cloudject points}
        getMinimumDistances(pCloudject, distances);
        
        // From distances to averaged scores
        scores.resize(distances.size());
        for (int v = 0; v < distances.size(); v++)
        {
            float scoreAcc = 0.f;
            int inliers = 0;
            
            for (int k = 0; k < distances[v].size(); k++)
            {
                float score = distToScore(distances[v][k]);
                if (score > m_PointRejectionThresh)
                {
                    scoreAcc += score;
                    inliers ++;
                }
            }
            
            scores[v] = ((inliers > 0) ? (scoreAcc / inliers) : 0.f); // do not count on rejections
        }
    }
    
    //    float getScore(typename LFCloudject<PointT,SignatureT>::Ptr pCloudject)
    //    {
    //        vector<float> penalizedScores (pCloudject->getNumOfViews());
    //
    //        float pnlScoreAcc = 0.f;
    //        float invDistAcc = 0.f;
    //
    //        for (int v = 0; v < pCloudject->getNumOfViews(); v++)
    //        {
    //            // Compute score and penalty
    //
    //            float score = matchView(pCloudject->getDescription(v));
    //
    //            float penalty = 1.f;
    //			if (getPenalty() == 1)
    //			{
    //                float avg = CloudjectModelBase<PointT,SignatureT>::averageNumOfPointsInModels();
    //				float ratio = pCloudject->getNumOfPointsInView(v) / avg;
    //				float x = (ratio <= 1.f) ? ratio : (1.f / ratio);
    //                float b = m_SigmaPenaltyThresh;
    //
    //                penalty *= 1.f / ( 1.f + expf( -(x-0.5f) * b) );
    //			}
    //			else if (getPenalty() == 2)
    //			{
    //                float avg = CloudjectModelBase<PointT,SignatureT>::averageMedianDistanceToCentroids();
    //
    //				float ratio = (pCloudject->medianDistToCentroidInView(v) / avg);
    //                float x = (ratio <= 1.f) ? ratio : (1.f / ratio);
    //                float b = m_SigmaPenaltyThresh;
    //
    //                penalty *= 1.f / ( 1.f + expf( -(x-0.5f) * b) );
    //
    //				//penalty *= (1.f / (m_SigmaPenaltyThresh * sqrtf(2.f * 3.14159))) * expf(-0.5f * powf(diff/m_SigmaPenaltyThresh, 2));
    //			}
    //
    //            pcl::PointXYZ pos = pCloudject->getPosition(v);
    //            float invDist = 1.f / sqrt(pow(pos.x,2) + pow(pos.y,2) + pow(pos.z,2));
    //
    //            pnlScoreAcc += invDist * (score * penalty);
    //            invDistAcc += invDist;
    //        }
    //
    //        return pnlScoreAcc / invDistAcc;
    //    }
    
protected:
	// Returns the score of matching a description of a certain cloudject's view against the model views' descriptions
    
    void getMinimumNonOverlapingDistancesToDescription(DescriptionPtr pDescription, vector<float>& distances)
    {
        distances.clear();
        
        // Structure to not re-match against a model point
        vector<vector<bool> > matches (m_ViewsDescriptions.size());
		for (int v = 0; v < m_ViewsDescriptions.size(); v++)
			matches[v].resize(m_ViewsDescriptions[v]->points.size(), false);
        
        // Count the positive matches to each model
        vector<int> numOfMatches (m_ViewsDescriptions.size(), 0);
        
        distances.resize(pDescription->size());
        for (int i = 0; i < pDescription->size(); i++) // test view
		{
            float minDistVal = std::numeric_limits<float>::max();
            int minDistViewIdx = -1;
            int minDistPointIdx = -1;
            
            for (int v = 0; v < m_ViewsDescriptions.size(); v++) // views of the model
			{
                // No points left to be matched in this model view, no need to go further with it
                if (numOfMatches[v] < m_ViewsDescriptions[v]->size())
                {
                    for (int j = 0; j < m_ViewsDescriptions[v]->size(); j++)
                    {
						if (!matches[v][j])
						{
                            //                        float dist = euclideanDistanceSignatures( pDescription->points[i], m_ViewsDescriptions[v]->points[j]);
							float dist = chisquareDistanceSignatures( pDescription->points[i], m_ViewsDescriptions[v]->points[j]);
                            //                        float dist = battacharyyaDistanceSignatures( pDescription->points[i], m_ViewsDescriptions[v]->points[j]);
							if (dist < minDistVal)
							{
								minDistViewIdx  = v;
								minDistPointIdx = j;
								minDistVal = dist;
							}
						}
                    }
                }
            }
            
			if (minDistViewIdx > 0 && minDistPointIdx > 0)
			{
				distances[i] = minDistVal;
				matches[minDistViewIdx][minDistPointIdx] = true;
				numOfMatches[minDistViewIdx] ++;
			}
        }
    }
    
    void getMinimumDistancesToDescription(DescriptionPtr pDescription, vector<float>& distances)
    {
        distances.clear();
        
        distances.resize(pDescription->size());
        for (int i = 0; i < pDescription->size(); i++) // test view
		{
            float minDistVal = std::numeric_limits<float>::max();

            for (int v = 0; v < m_ViewsDescriptions.size(); v++) // views of the model
			{
                for (int j = 0; j < m_ViewsDescriptions[v]->size(); j++)
                {
//                        float dist = euclideanDistanceSignatures( pDescription->points[i], m_ViewsDescriptions[v]->points[j]);
                    float dist = chisquareDistanceSignatures( pDescription->points[i], m_ViewsDescriptions[v]->points[j]);
//                        float dist = battacharyyaDistanceSignatures( pDescription->points[i], m_ViewsDescriptions[v]->points[j]);
                    if (dist < minDistVal)
                    {
                        minDistVal = dist;
                    }
                }
            }
            
            distances[i] = minDistVal;
        }
    }
    
	// Returns the battacharyya distance between two fpfh signatures, which are actually histograms.
	// This is a normalized [0,1] distance
	float battacharyyaDistanceSignatures(SignatureT& s1, SignatureT& s2)
	{
		float accSqProd = 0.f;
		float accS1 = 0.f;
		float accS2 = 0.f;
        
        int B = sizeof(s1.histogram) / sizeof(s1.histogram[0]);
		for (int b = 0; b < B;  b++)
		{
			accSqProd += sqrt(s1.histogram[b] * s2.histogram[b]);
			accS1 += s1.histogram[b];
			accS2 += s2.histogram[b];
		}
        
		float f = 1.f / sqrt((accS1/B) * (accS2/B) * (B*B));
        
		return sqrt(1.f - f * accSqProd);
	}
    
    float chisquareDistanceSignatures(SignatureT& s1, SignatureT& s2)
	{
		float accS1, accS2;
        accS1 = accS2 = 0.f;
        
        int B = sizeof(s1.histogram) / sizeof(s1.histogram[0]);
		for (int b = 0; b < B; b++)
		{
			accS1 += s1.histogram[b];
			accS2 += s2.histogram[b];
		}
        
        float acc = 0.f;
        for (int b = 0; b < B; b++)
        {
            float val1 = (accS1 > 0) ? (s1.histogram[b] / accS1) : 0;
            float val2 = (accS2 > 0) ? (s2.histogram[b] / accS2) : 0;
            
            float m = (val1 + val2);
            if (m > 0)
                acc += (powf(val1 - val2, 2) / m);
        }
        
		return 0.5f * acc;
	}
    
	// Returns the euclidean distance between two fpfh signatures, which are actually histograms
	float euclideanDistanceSignatures(SignatureT s1, SignatureT s2)
	{
        float accS1 = 0.f;
		float accS2 = 0.f;
        
        int B = sizeof(s1.histogram) / sizeof(s1.histogram[0]);
		for (int b = 0; b < B;  b++)
		{
			accS1 += s1.histogram[b];
			accS2 += s2.histogram[b];
		}
        
		float acc = 0.f;
        
		for (int b = 0; b < B; b++)
		{
			acc += powf(s1.histogram[b]/accS1 - s2.histogram[b]/accS2, 2.0);
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
    
	int getPenalty()
	{
		return m_Penalty;
	}
    
	//
	// Protected members
	//
    
	// The descriptions of the different views
	vector<DescriptionPtr>		m_ViewsDescriptions;
	// A valid best correspondence should be a distance below it (experimentally selected)
	float       m_PointRejectionThresh;
	float       m_RatioRejectionThresh;
	float		m_SigmaPenaltyThresh;
    
	int			m_Penalty;
	enum		Penalty { None, NumOfPoints, MedianDistToCentroid };
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
	typedef pcl::PointCloud<pcl::FPFHSignature33> Description;
	typedef pcl::PointCloud<pcl::FPFHSignature33>::Ptr DescriptionPtr;
	typedef pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
    typedef pcl::search::KdTree<PointT> KdTree;
    typedef typename pcl::search::KdTree<PointT>::Ptr KdTreePtr;
    
//	typedef LFCloudject<PointT,pcl::FPFHSignature33> LFCloudject;
    
public:
    LFCloudjectModel(float leafSize = 0.f)
    : LFCloudjectModelBase<PointT,pcl::FPFHSignature33>(leafSize)
    {}
    
	LFCloudjectModel(int ID, string name, float leafSize = 0.0, int penalty = 1, float pointRejectionThresh = 1.0, float ratioRejectionThresh = 1.0, float sigmaPenaltyThresh = 0.1)
    : LFCloudjectModelBase<PointT,pcl::FPFHSignature33>(ID, name, leafSize, penalty, pointRejectionThresh, ratioRejectionThresh, sigmaPenaltyThresh)
	{}
    
    LFCloudjectModel(int ID, string name, vector<PointCloudPtr> views, float leafSize = 0.0, int penalty = 1, float pointRejectionThresh = 1.0, float ratioRejectionThresh = 1.0, float sigmaPenaltyThresh = 0.1)
    : LFCloudjectModelBase<PointT,pcl::FPFHSignature33>(ID, name, views, leafSize, penalty, pointRejectionThresh, ratioRejectionThresh, sigmaPenaltyThresh)
	{}
    
    LFCloudjectModel(const LFCloudjectModel& rhs)
    : LFCloudjectModelBase<PointT,pcl::FPFHSignature33>(rhs)
    {
        *this = rhs;
    }
    
    //	virtual ~LFCloudjectModel() {}
    
    LFCloudjectModel& operator=(const LFCloudjectModel& rhs)
    {
        return *this;
    }
    
	// Describe all the model views
	void describe(float normalRadius, float fpfhRadius, float leafSize = 0.f)
	{
		for (int i = 0; i < LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::getNumOfViews(); i++)
		{
			DescriptionPtr pDescription (new Description);
            PointCloudPtr view = LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::getView(i);
            if (leafSize > LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::m_LeafSize)
                describeView(view, leafSize, normalRadius, fpfhRadius, *pDescription);
            else
                describeView(view, normalRadius, fpfhRadius, *pDescription);
			LFCloudjectModelBase<PointT,pcl::FPFHSignature33>::addViewDescription(pDescription);
		}
	}
    
    typedef boost::shared_ptr<LFCloudjectModel<PointT,pcl::FPFHSignature33> > Ptr;
    
private:
    
	// Compute the description of a view, performing
	// a prior downsampling to speed up the process
	void describeView(PointCloudPtr pView, float leafSize, float normalRadius, float fpfhRadius, Description& Description)
	{
		PointCloudPtr pViewF (new PointCloud);
        
		pcl::ApproximateVoxelGrid<PointT> avg;
		avg.setInputCloud(pView);
		avg.setLeafSize(leafSize, leafSize, leafSize);
		avg.filter(*pViewF);
        
		describeView(pViewF, normalRadius, fpfhRadius, Description);
	}
    
    
	// Compute the description of a view, actually
	void describeView(PointCloudPtr pView,
					  float normalRadius, float fpfhRadius,
					  Description& Description)
	{
		//
		// Normals preprocess
		//
        
		// Create the normal estimation class, and pass the input dataset to it
		pcl::NormalEstimation<PointT, pcl::Normal> ne;
		ne.setInputCloud (pView);
        
		// Create an empty kdtree representation, and pass it to the normal estimation object.
		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
		KdTreePtr tree (new KdTree);
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
		tree = KdTreePtr(new KdTree);
		fpfh.setSearchMethod (tree);
        
		// Output datasets
		// * initialize outside
        
		// Use all neighbors in a sphere of radius 5cm
		// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
		fpfh.setRadiusSearch (fpfhRadius);
        
		// Compute the features
		fpfh.compute (Description);
	}
};

// Partially specialized template
template<>
class LFCloudjectModel<pcl::PointXYZRGB, pcl::PFHRGBSignature250> : public LFCloudjectModelBase<pcl::PointXYZRGB, pcl::PFHRGBSignature250>
{
    
    typedef pcl::PointXYZRGB PointT;
	typedef pcl::PointCloud<pcl::PFHRGBSignature250> Description;
	typedef pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr DescriptionPtr;
	typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
	typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudPtr;
    typedef pcl::search::KdTree<pcl::PointXYZRGB> KdTree;
    typedef pcl::search::KdTree<pcl::PointXYZRGB>::Ptr KdTreePtr;
    
//	typedef LFCloudject<PointT,pcl::PFHRGBSignature250> LFCloudject;
    
public:
    
    LFCloudjectModel(float leafSize = 0.f)
    : LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>(leafSize)
    {}
    
	LFCloudjectModel(int ID, string name, float leafSize = 0.0, int penalty = 1, float pointRejectionThresh = 1.0, float ratioRejectionThresh = 1.0, float sigmaPenaltyThresh = 0.1)
    : LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>(ID, name, leafSize, penalty, pointRejectionThresh, ratioRejectionThresh, sigmaPenaltyThresh)
	{}
    
    LFCloudjectModel(int ID, string name, vector<PointCloudPtr> views, float leafSize = 0.0, int penalty = 1, float pointRejectionThresh = 1.0, float ratioRejectionThresh = 1.0, float sigmaPenaltyThresh = 0.1)
    : LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>(ID, name, views, leafSize, penalty, pointRejectionThresh, ratioRejectionThresh, sigmaPenaltyThresh)
	{}
    
    LFCloudjectModel(const LFCloudjectModel& rhs)
    : LFCloudjectModelBase<PointT,pcl::PFHRGBSignature250>(rhs)
    {
        *this = rhs;
    }
    
    //	virtual ~LFCloudjectModel() {}
    
    LFCloudjectModel& operator=(const LFCloudjectModel& rhs)
    {
        return *this;
    }
    
	// Describe all the model views
	void describe(float normalRadius, float pfhrgbRadius, float leafSize = 0.f)
	{
		for (int i = 0; i < getNumOfViews(); i++)
		{
			DescriptionPtr pDescription (new Description);
            PointCloudPtr view = getView(i);
            if (leafSize > m_LeafSize)
                describeView(view, leafSize, normalRadius, pfhrgbRadius, *pDescription);
            else
                describeView(view, normalRadius, pfhrgbRadius, *pDescription);
			addViewDescription(pDescription);
		}
	}
    
    typedef boost::shared_ptr<LFCloudjectModel<pcl::PointXYZRGB,pcl::PFHRGBSignature250> > Ptr;
    
private:
    
	// Compute the description of a view, performing
	// a prior downsampling to speed up the process
	void describeView(PointCloudPtr pView, float leafSize, float normalRadius, float pfhrgbRadius, Description& Description)
	{
		PointCloudPtr pViewF (new PointCloud);
        
		pcl::ApproximateVoxelGrid<PointT> avg;
		avg.setInputCloud(pView);
		avg.setLeafSize(leafSize, leafSize, leafSize);
		avg.filter(*pViewF);
        
		describeView(pViewF, normalRadius, pfhrgbRadius, Description);
	}
    
    
	// Compute the description of a view, actually
	void describeView(PointCloudPtr pView, float normalRadius, float pfhrgbRadius,
					  Description& Description)
	{
		//
		// Normals preprocess
		//
        
		// Create the normal estimation class, and pass the input dataset to it
		pcl::NormalEstimation<PointT, pcl::Normal> ne;
		ne.setInputCloud (pView);
        
		// Create an empty kdtree representation, and pass it to the normal estimation object.
		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
		KdTreePtr tree (new KdTree);
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
        
		// Create an empty kdtree representation, and pass it to the PFHRGB estimation object.
		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
		tree = KdTreePtr(new KdTree);
		pfhrgb.setSearchMethod (tree);
        
		// Output datasets
		// * initialize outside
        
		// Use all neighbors in a sphere of radius 5cm
		// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
		pfhrgb.setRadiusSearch (pfhrgbRadius);
        
		// Compute the features
		pfhrgb.compute (Description);
	}
};

// Explicit template instantation
template class CloudjectModelBase<pcl::PointXYZ,pcl::FPFHSignature33>;
template class CloudjectModelBase<pcl::PointXYZ,pcl::PFHRGBSignature250>;
template class CloudjectModelBase<pcl::PointXYZRGB,pcl::FPFHSignature33>;
template class CloudjectModelBase<pcl::PointXYZRGB,pcl::PFHRGBSignature250>;

template class LFCloudjectModelBase<pcl::PointXYZ,pcl::FPFHSignature33>;
template class LFCloudjectModelBase<pcl::PointXYZ,pcl::PFHRGBSignature250>;
template class LFCloudjectModelBase<pcl::PointXYZRGB,pcl::FPFHSignature33>;
template class LFCloudjectModelBase<pcl::PointXYZRGB,pcl::PFHRGBSignature250>;

template class LFCloudjectModel<pcl::PointXYZ,pcl::FPFHSignature33>;
template class LFCloudjectModel<pcl::PointXYZRGB,pcl::FPFHSignature33>;
