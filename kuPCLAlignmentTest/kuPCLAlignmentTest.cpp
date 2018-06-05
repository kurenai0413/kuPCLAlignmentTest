#define _SCL_SECURE_NO_WARNINGS

#include <iostream>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
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
void CalculateSHOTDescriptors(pcl::PointCloud<pcl::PointXYZ>::Ptr pcd,
							  double res,
							  pcl::PointCloud<pcl::Normal>::Ptr normals,
							  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints,
							  pcl::PointCloud<pcl::SHOT352>::Ptr &decriptors);
void MatchKeypoints(pcl::PointCloud<pcl::SHOT352>::Ptr srcDes, pcl::PointCloud<pcl::SHOT352>::Ptr tarDes);

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
	pcl::PointCloud<pcl::SHOT352>::Ptr	srcDescriptors(new pcl::PointCloud<pcl::SHOT352>());
	pcl::PointCloud<pcl::SHOT352>::Ptr	tarDescriptors(new pcl::PointCloud<pcl::SHOT352>());
	pcl::CorrespondencesPtr				corres(new pcl::Correspondences());

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

	CalculateSHOTDescriptors(srcPCD, srcRes, srcNormals, srcKeyPts, srcDescriptors);
	CalculateSHOTDescriptors(tarPCD, tarRes, tarNormals, tarKeyPts, tarDescriptors);

	std::vector<float>	corresDists;
	std::vector<int>	corresIndices;
	std::vector<int>	srcIndices;

	pcl::KdTreeFLANN<pcl::SHOT352> matchSearch;   
	matchSearch.setInputCloud(tarDescriptors);
	for (size_t i = 0; i < srcDescriptors->size(); i++)
	{
		std::vector<int>   neighborIndice(1);
		std::vector<float> neighborSqrtDist(1);

		if (!pcl_isfinite(srcDescriptors->at(i).descriptor[0]))
		{
			continue;
		}

		int foundNeighbors = matchSearch.nearestKSearch(srcDescriptors->at(i), 1, neighborIndice, neighborSqrtDist);
	
		if (foundNeighbors == 1 && neighborSqrtDist[0] < 0.25f) // if find correspondence and distance below 0.25
		{ 
			pcl::Correspondence corr(neighborIndice[0], static_cast<int> (i), neighborSqrtDist[0]);
			corres->push_back(corr);
			corresDists.push_back(neighborSqrtDist[0]);
			corresIndices.push_back(neighborIndice[0]);
			srcIndices.push_back(i);
		}
	}

	file.open("NeighborInfo.txt", std::ios::out);
	for (int i = 0; i < corres->size(); i++)
	{
		file << i << ": " << "Src idx: " << srcIndices[i]
			 << ", Tar idx: " << corresIndices[i] << ", dist: " << corresDists[i] << std::endl;
	}
	file.close();

	std::cout << "Correspondence num: " << corres->size() << std::endl;

	int matchPtsNum = corres->size();
	std::vector<int>	srcMatchPtsIdx;
	std::vector<int>	tarMatchPtsIdx;

	for (int i = 0; i < matchPtsNum; i++)
	{
		srcMatchPtsIdx.push_back(srcKeypointsIndices->indices[srcIndices[i]]);
		tarMatchPtsIdx.push_back(tarKeypointsIndices->indices[corresIndices[i]]);
	}

	file.open("MatchPts.txt", std::ios::out);
	for (int i = 0; i < matchPtsNum; i++)
	{
		file << srcMatchPtsIdx[i] << " " << srcPCD->points[srcMatchPtsIdx[i]].x <<
									 " " << srcPCD->points[srcMatchPtsIdx[i]].y <<
									 " " << srcPCD->points[srcMatchPtsIdx[i]].z << " "
			 << tarMatchPtsIdx[i] << " " << tarPCD->points[tarMatchPtsIdx[i]].x <<
									 " " << tarPCD->points[tarMatchPtsIdx[i]].x <<
									 " " << tarPCD->points[tarMatchPtsIdx[i]].x << std::endl;
	}
	file.close();

	//WriteKeypointsToFile("Source_Keypoints.txt", srcPCD, srcKeypointsIndices);
	//WriteKeypointsToFile("Target_Keypoints.txt", tarPCD, tarKeypointsIndices);

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
	normalEst.setNumberOfThreads(16);
	normalEst.compute(*normals);
}

void CalculateSHOTDescriptors(pcl::PointCloud<pcl::PointXYZ>::Ptr pcd, double res, pcl::PointCloud<pcl::Normal>::Ptr normals, pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints, pcl::PointCloud<pcl::SHOT352>::Ptr &descriptors)
{
	pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> descriptorEst;

	descriptorEst.setRadiusSearch(10 * res);
	descriptorEst.setInputCloud(keypoints);
	descriptorEst.setInputNormals(normals);
	descriptorEst.setSearchSurface(pcd);
	descriptorEst.setNumberOfThreads(16);
	descriptorEst.compute(*descriptors);


}

void MatchKeypoints(pcl::PointCloud<pcl::SHOT352>::Ptr srcDes, pcl::PointCloud<pcl::SHOT352>::Ptr tarDes)
{

}
