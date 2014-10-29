#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/features/normal_3d.h>

using namespace std;

template<typename PointT, typename SignatureT>
class CloudjectBase
{
	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename PointCloud::Ptr PointCloudPtr;
    
public:
	CloudjectBase(float leafSize = 0.f) 
		: m_ID(0), m_Name(""), m_LeafSize(leafSize)
	{ 
	}
    
    CloudjectBase(PointCloudPtr pView, float leafSize = 0.f)
        : m_ID(0), m_Name(""), m_LeafSize(leafSize)
	{
        addView(pView);
	}
    
    CloudjectBase(string viewFilepath, float leafSize = 0.f)
        : m_ID(0), m_Name(""), m_LeafSize(leafSize)
	{
        PointCloudPtr pView (new PointCloud);
        
        pcl::PCDReader reader;
        reader.read(viewFilepath, *pView);
        
        addView(pView);
	}
    
    CloudjectBase(vector<PointCloudPtr> views, float leafSize = 0.f)
        : m_ID(0), m_Name(""), m_LeafSize(leafSize)
	{
        for (int v = 0; v < views.size(); v++)
		{
			addView(views[v]);
		}
	}
    
    CloudjectBase(vector<string> viewsFilepaths, float leafSize = 0.f)
		: m_ID(0), m_Name(""), m_LeafSize(leafSize)
	{
		for (int v = 0; v < viewsFilepaths.size(); v++)
		{
			PointCloudPtr pView (new PointCloud);
        
			pcl::PCDReader reader;
			reader.read(viewsFilepaths[v], *pView);

			addView(pView);
		}
	}
    
    CloudjectBase(vector<pair<int,PointCloudPtr> > views, float leafSize = 0.f)
		: m_ID(0), m_Name(""), m_LeafSize(leafSize)
	{
        for (int v = 0; v < views.size(); v++)
		{
			addView(views[v].second, views[v].first);
		}
	}

	CloudjectBase(const CloudjectBase& cloudject)
	{
        *this = cloudject;
	}

    CloudjectBase& operator=(const CloudjectBase& cloudject)
    {
        if (this != &cloudject)
        {
            m_ID	= cloudject.m_ID;
            m_Name = cloudject.m_Name;
            m_LeafSize = cloudject.m_LeafSize;
            
            m_ViewIDs = cloudject.m_ViewIDs;
            m_OriginalViews = cloudject.m_OriginalViews;
            m_Views = cloudject.m_Views;
            m_Positions	= cloudject.m_Positions;
            m_MedianDists = cloudject.m_MedianDists;
        }
        
        return *this;
    }
    
	void addView(PointCloudPtr pView, int id = -1)
	{
		// Get the new view ID
		int vid;
		if (id < 0) // Default argument
			vid = m_Views.size(); // auto-incremental id
		else 
			vid = id; // passed-by one

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
		m_ViewIDs.push_back(vid);
		m_OriginalViews.push_back(pOriginalView);
		m_Views.push_back(pView);
        m_Positions.push_back(centroidPt);
        m_MedianDists.push_back(medianDist);
	}
    
	int getID() { return m_ID; }
	void setID(int ID) { m_ID = ID; }
    
    string getName() { return m_Name; }
    void setName(string name) { m_Name = name; }

	float getDownsamplingSize() { return m_LeafSize; }
    
    pcl::PointXYZ getPosition(int v) const
	{
		return m_Positions[v];
	}

	int getNumOfPointsInOriginalView(int v)
	{
		return m_OriginalViews[v]->points.size();
	}

	int getNumOfPointsInView(int v)
	{
		return m_Views[v]->points.size();
	}

	float medianDistToCentroidInView(int v)
	{
		return m_MedianDists[v];
	}

    int getNumOfViews() const
    {
        return m_Views.size();
    }
    
	PointCloudPtr getView(int v) const
	{
		return m_Views[v];
	}
    
    vector<int> getViewIDs() const
	{
		return m_ViewIDs;
	}
    
    vector<pcl::PointXYZ> getPositions() const
    {
        return m_Positions;
    }
    
    template<typename PointT1, typename PointT2>
	float euclideanDistance(PointT1 p1, PointT2 p2)
	{
		return sqrt(powf(p1.x - p2.x, 2) + powf(p1.y - p2.y, 2) + powf(p1.z - p2.z, 2));
	}
    
	float medianDistanceToCentroid(PointCloudPtr pCloud, pcl::PointXYZ centroid)
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

        int medianIdx = distances.size() / 2;
		return distances[medianIdx];
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

	// Methods

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

	// Members

	int m_ID;
    string m_Name;
    float m_LeafSize;

	vector<int> m_ViewIDs;
	vector<PointCloudPtr> m_OriginalViews;
	vector<PointCloudPtr> m_Views;
	vector<pcl::PointXYZ> m_Positions;
	vector<float> m_MedianDists;
};


// Locally Featured (LF) Cloudject
template<typename PointT, typename SignatureT>
class LFCloudjectBase : public CloudjectBase<PointT,SignatureT>
{
	typedef typename pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
	typedef typename pcl::PointCloud<SignatureT> Description;
	typedef typename pcl::PointCloud<SignatureT>::Ptr DescriptionPtr;

public:
	LFCloudjectBase(float leafSize = 0.f)
        : CloudjectBase<PointT,SignatureT>(leafSize)
    {
    
    }

    LFCloudjectBase(PointCloudPtr pView, float leafSize = 0.f)
        : CloudjectBase<PointT,SignatureT>(pView, leafSize)
    {
        m_Descriptions.resize(1);
    }
    
    LFCloudjectBase(string viewFilepath, float leafSize = 0.f)
        : CloudjectBase<PointT,SignatureT>(viewFilepath, leafSize)
    {
        m_Descriptions.resize(1);
    }
    
	LFCloudjectBase(vector<PointCloudPtr> views, float leafSize = 0.f)
		: CloudjectBase<PointT,SignatureT>(views, leafSize)
    {
        m_Descriptions.resize(views.size());
    }
    
	LFCloudjectBase(vector<string> viewsFilePaths, float leafSize = 0.f)
		: CloudjectBase<PointT,SignatureT>(viewsFilePaths, leafSize)
    {
        m_Descriptions.resize(viewsFilePaths.size());
    }
    
    LFCloudjectBase(vector<pair<int,PointCloudPtr> > views, float leafSize = 0.f)
    : CloudjectBase<PointT,SignatureT>(views, leafSize)
    {
        m_Descriptions.resize(views.size());
    }

	LFCloudjectBase(const LFCloudjectBase<PointT,SignatureT>& rhs)
		: CloudjectBase<PointT,SignatureT>(rhs)
	{
        *this = rhs;
	}

//	virtual ~LFCloudjectBase() {}
    
    LFCloudjectBase& operator=(const LFCloudjectBase<PointT,SignatureT>& rhs)
    {
        if (this != &rhs)
        {
            m_Descriptions = rhs.m_Descriptions;
        }
        
        return *this;
    }

	void setDescriptions(vector<DescriptionPtr> descriptions)
	{
		m_Descriptions = descriptions;
	}

	DescriptionPtr getDescription(int v)
	{ 
		return m_Descriptions[v];
	}
    
protected:
	vector<DescriptionPtr> m_Descriptions;
};


// Locally Featured (LF) Cloudject
template<typename PointT, typename SignatureT>
class LFCloudject : public LFCloudjectBase<PointT,SignatureT>
{
public:

    typedef boost::shared_ptr<LFCloudject<PointT,SignatureT> > Ptr;

};


template<typename PointT>
class LFCloudject<PointT, pcl::FPFHSignature33> : public LFCloudjectBase<PointT, pcl::FPFHSignature33>
{
    typedef pcl::PointCloud<PointT> PointCloud;
	typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
    typedef pcl::search::KdTree<PointT> KdTree;
	typedef typename pcl::search::KdTree<PointT>::Ptr KdTreePtr;

public:
	LFCloudject(float leafSize = 0.f)
		: LFCloudjectBase<PointT,pcl::FPFHSignature33>(leafSize)
    { }

    LFCloudject(PointCloudPtr pView, float leafSize = 0.f)
        : LFCloudjectBase<PointT,pcl::FPFHSignature33>(pView, leafSize) { }
    
    LFCloudject(string viewFilepath, float leafSize = 0.f)
        : LFCloudjectBase<PointT,pcl::FPFHSignature33>(viewFilepath, leafSize) { }
    
	LFCloudject(vector<PointCloudPtr> views, float leafSize = 0.f)
		: LFCloudjectBase<PointT,pcl::FPFHSignature33>(views, leafSize) { }

	LFCloudject(vector<string> viewFilePaths, float leafSize = 0.f)
        : LFCloudjectBase<PointT,pcl::FPFHSignature33>(viewFilePaths, leafSize) { }
    
	LFCloudject(vector<pair<int,PointCloudPtr> > views, float leafSize = 0.f)
        : LFCloudjectBase<PointT,pcl::FPFHSignature33>(views, leafSize) { }

	LFCloudject(const LFCloudject<PointT,pcl::FPFHSignature33>& rhs)
		: LFCloudjectBase<PointT,pcl::FPFHSignature33>(rhs)
    {
        *this = rhs;
    }

//	virtual ~LFCloudject() {}

	//
	// Methods
	//
    
    LFCloudject& operator=(const LFCloudject<PointT,pcl::FPFHSignature33>& rhs)
    {
        return *this;
    }
  
	void describe(float normalRadius, float fpfhRadius, float leafSize = 0.f)
	{
        for (int i = 0; i < LFCloudjectBase<PointT, pcl::FPFHSignature33>::getNumOfViews(); i++)
        {
            PointCloudPtr pView = LFCloudjectBase<PointT, pcl::FPFHSignature33>::getView(i);
            if (!pView->empty())
            {
                LFCloudjectBase<PointT,pcl::FPFHSignature33>::m_Descriptions[i] = pcl::PointCloud<pcl::FPFHSignature33>::Ptr (new pcl::PointCloud<pcl::FPFHSignature33>);
                describeView( pView, normalRadius, fpfhRadius, *(LFCloudjectBase<PointT,pcl::FPFHSignature33>::m_Descriptions[i]));
            }
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
		KdTreePtr tree (new KdTree());
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
		tree = KdTreePtr(new KdTree());
		fpfh.setSearchMethod (tree);

		// Output datasets
		// * initialize outside

		// Use all neighbors in a sphere of radius 5cm
		// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
		fpfh.setRadiusSearch (fpfhRadius);

		// Compute the features
		fpfh.compute (descriptor);
	}
    
    typedef boost::shared_ptr<LFCloudject<PointT,pcl::FPFHSignature33> > Ptr;
    
//private:
//	void downsample(PointCloud& cloud, float leafSize, PointCloud& filteredCloud)
//	{ LFCloudjectBase<PointT,pcl::FPFHSignature33>::downsample(cloud, leafSize, filteredCloud); }
};


template<>
class LFCloudject<pcl::PointXYZRGB, pcl::PFHRGBSignature250> : public LFCloudjectBase<pcl::PointXYZRGB, pcl::PFHRGBSignature250>
{
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
	typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudPtr;
    typedef pcl::search::KdTree<pcl::PointXYZRGB> KdTree;
	typedef pcl::search::KdTree<pcl::PointXYZRGB>::Ptr KdTreePtr;

public:
	LFCloudject(float leafSize = 0.f)
		: LFCloudjectBase<PointT,pcl::PFHRGBSignature250>(leafSize) { }

    LFCloudject(PointCloudPtr pView, float leafSize = 0.f)
        : LFCloudjectBase<PointT,pcl::PFHRGBSignature250>(pView, leafSize) { }
    
    LFCloudject(string viewFilepath, float leafSize = 0.f)
        : LFCloudjectBase<PointT,pcl::PFHRGBSignature250>(viewFilepath, leafSize) { }
    
	LFCloudject(vector<PointCloudPtr> views, float leafSize = 0.f)
        : LFCloudjectBase<PointT,pcl::PFHRGBSignature250>(views, leafSize) { }
    
    LFCloudject(vector<string> viewsFilePaths, float leafSize = 0.f)
        : LFCloudjectBase<PointT,pcl::PFHRGBSignature250>(viewsFilePaths, leafSize) { }
    
	LFCloudject(vector<pair<int,PointCloudPtr> > views, float leafSize = 0.f)
        : LFCloudjectBase<PointT,pcl::PFHRGBSignature250>(views, leafSize) { }
    
	LFCloudject(const LFCloudject<PointT,pcl::PFHRGBSignature250>& cloudject) 
		: LFCloudjectBase<PointT,pcl::PFHRGBSignature250>(cloudject) { }

    
    //	virtual ~LFCloudject() {}
    
	//
	// Methods
	//

	void describe(float normalRadius, float pfhrgbRadius)
	{
        for (int i = 0; i < LFCloudjectBase<PointT,pcl::PFHRGBSignature250>::getNumOfViews(); i++)
        {
            PointCloudPtr pView = getView(i);
            if (!pView->empty())
            {
                m_Descriptions[i] = pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr (new  pcl::PointCloud<pcl::PFHRGBSignature250>);
                describeView(pView, normalRadius, pfhrgbRadius, *(m_Descriptions[i]));
            }
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
		KdTreePtr tree (new KdTree());
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

    typedef boost::shared_ptr<LFCloudject<pcl::PointXYZRGB,pcl::PFHRGBSignature250> > Ptr;
};
