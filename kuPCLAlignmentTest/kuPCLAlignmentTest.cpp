#define _SCL_SECURE_NO_WARNINGS

#include <iostream>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/iss_3d.h>
#include <ANN.h>

double
ComputeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	double resolution = 0.0;
	int numberOfPoints = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> squaredDistances(2);
	pcl::search::KdTree<pcl::PointXYZ> tree;
	tree.setInputCloud(cloud);

	// (by Kurenai) Compute neareset neighbor RMS as surface resolution
	for (size_t i = 0; i < cloud->size(); ++i)
	{
		if (!pcl_isfinite((*cloud)[i].x))
			continue;

		// Considering the second neighbor since the first is the point itself.
		nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
		if (nres == 2)
		{
			resolution += sqrt(squaredDistances[1]);
			++numberOfPoints;
		}
	}
	if (numberOfPoints != 0)
		resolution /= numberOfPoints;

	return resolution;
}

void UniformSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr ptsSrc, pcl::PointCloud<pcl::PointXYZ>::Ptr ptsTar, int ptsNum, int sampleNum)
{
	if (ptsSrc->size() != 0)
	{
		srand(time(NULL));

		int level = ptsNum / sampleNum;

		for (int i = 0; i < sampleNum; i++)
		{
			int rndIdx = rand() % level;
			ptsTar->insert(ptsTar->end(), ptsSrc->points[level * i + rndIdx]);
		}
	}
}

void LoadPCDFromFile(char * filename, pcl::PointCloud<pcl::PointXYZ>::Ptr ptCloud, int &ptsNum)
{
	FILE * fp;
	pcl::PointXYZ ptTemp;

	if (ptCloud->width != 0)
		ptCloud->clear();

	errno_t err = fopen_s(&fp, filename, "r");
	while (fscanf_s(fp, "%f %f %f\n", &ptTemp.x, &ptTemp.y, &ptTemp.z) != EOF)
	{
		ptCloud->insert(ptCloud->end(), ptTemp);
	}
	fclose(fp);

	ptsNum = ptCloud->width;
}

void ExtractISSKeypoints(pcl::PointCloud<pcl::PointXYZ>::Ptr pcd, double resolution,
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPts,
	pcl::PointIndicesConstPtr &keypointIndices);
void WriteKeypointsToFile(char * filename, pcl::PointCloud<pcl::PointXYZ>::Ptr pcd,
	pcl::PointIndicesConstPtr keypointIndices);
void EstimateSurfaceNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr pcd,
	pcl::PointCloud<pcl::Normal>::Ptr &normals);

void main()
{
	// Objects for storing the point cloud and the keypoints.
	std::fstream file;

	pcl::PointCloud<pcl::PointXYZ>::Ptr srcPCD(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr tarPCD(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr	srcNormals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr	tarNormals(new pcl::PointCloud<pcl::Normal>);
	int	srcPtsNum, tarPtsNum;

	pcl::PointCloud<pcl::PointXYZ>::Ptr srcKeyPts(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr tarKeyPts(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointIndicesConstPtr			srcKeypointsIndices;
	pcl::PointIndicesConstPtr			tarKeypointsIndices;
	pcl::PointCloud<pcl::SHOT352>		srcDescriptors;
	pcl::PointCloud<pcl::SHOT352>		tarDesciprtors;

	LoadPCDFromFile("SampledRealSurface.txt", srcPCD, srcPtsNum);
	std::cout << "Source point num = " << srcPtsNum << std::endl;
	LoadPCDFromFile("CroppedCTSurface.txt", tarPCD, tarPtsNum);
	std::cout << "Target point num = " << tarPtsNum << std::endl;
	double srcRes = ComputeCloudResolution(srcPCD);
	std::cout << "Source resolution = " << srcRes << std::endl;
	double tarRes = ComputeCloudResolution(tarPCD);
	std::cout << "Target resolution = " << tarRes << std::endl;

	std::cout << "Load pts complete." << std::endl;

	EstimateSurfaceNormals(srcPCD, srcNormals);
	EstimateSurfaceNormals(tarPCD, tarNormals);

	std::cout << srcNormals->size() << std::endl;

	file.open("srcNormals.txt", std::ios::out);
	for (int i = 0; i < srcNormals->size(); i++)
	{
		file << srcNormals->points[i].normal_x << " "
			<< srcNormals->points[i].normal_y << " "
			<< srcNormals->points[i].normal_z << std::endl;
	}
	file.close();

	ExtractISSKeypoints(srcPCD, srcRes, srcKeyPts, srcKeypointsIndices);
	ExtractISSKeypoints(tarPCD, tarRes, tarKeyPts, tarKeypointsIndices);

	WriteKeypointsToFile("Source_Keypoints.txt", srcPCD, srcKeypointsIndices);
	WriteKeypointsToFile("Target_Keypoints.txt", tarPCD, tarKeypointsIndices);

	file.open("srcKeypts.txt", std::ios::out);
	for (int i = 0; i < srcKeyPts->size(); i++)
	{
		file << srcKeyPts->points[i].x << " "
			<< srcKeyPts->points[i].y << " "
			<< srcKeyPts->points[i].z << std::endl;
	}
	file.close();

	system("pause");
}

void ExtractISSKeypoints(pcl::PointCloud<pcl::PointXYZ>::Ptr pcd, double resolution, pcl::PointCloud<pcl::PointXYZ>::Ptr keyPts, pcl::PointIndicesConstPtr &keypointIndices)
{
	// ISS keypoint detector object.
	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> detector;
	detector.setInputCloud(pcd);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	detector.setSearchMethod(kdtree);

	// Set the radius of the spherical neighborhood used to compute the scatter matrix.
	detector.setSalientRadius(20 * resolution);
	// Set the radius for the application of the non maxima supression algorithm.
	detector.setNonMaxRadius(4 * resolution);
	// Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
	detector.setMinNeighbors(5);
	// Set the upper bound on the ratio between the second and the first eigenvalue.
	detector.setThreshold21(0.975);
	// Set the upper bound on the ratio between the third and the second eigenvalue.
	detector.setThreshold32(0.975);
	// Set the number of prpcessing threads to use. 0 sets it to automatic.
	detector.setNumberOfThreads(16);
	detector.compute(*keyPts);

	keypointIndices = detector.getKeypointsIndices();
	int keypointsNum = keypointIndices->indices.size();
	std::cout << "keypoints num: " << keypointsNum << std::endl;
}

void WriteKeypointsToFile(char * filename, pcl::PointCloud<pcl::PointXYZ>::Ptr pcd, pcl::PointIndicesConstPtr keypointIndices)
{
	int keypointsNum = keypointIndices->indices.size();

	std::fstream file;
	file.open(filename, std::ios::out);
	for (int i = 0; i < keypointsNum; i++)
	{
		int idx = keypointIndices->indices[i];
		file << pcd->points[idx].x << " "
			<< pcd->points[idx].y << " "
			<< pcd->points[idx].z << std::endl;
	}
	file.close();
}

void EstimateSurfaceNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr pcd, pcl::PointCloud<pcl::Normal>::Ptr & normals)
{
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEst;
	normalEst.setKSearch(10);
	normalEst.setInputCloud(pcd);
	normalEst.compute(*normals);
}
