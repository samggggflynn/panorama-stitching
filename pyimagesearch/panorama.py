# import the necessary packages
import numpy as np
import imutils
import cv2

class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		# 利用 is_cv3 判断是否在使用 opencv3
		self.isv3 = imutils.is_cv3(or_better=True)

	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		# 解压缩图像，然后检测关键点并提取来自它们的局部不变描述符
		(imageB, imageA) = images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)
		'''
		（我们再次假定只包含两个图像）。
		对图像列表的排序很重要：我们希望图像以从左到右的顺序提供。
		如果没有按此顺序提供图像，那么我们的代码仍将运行。
		但我们的输出全景图只包含一个图像，而不是两个图像。
		一旦读取图像列表，我们就调用DetectAndDescribe方法。
		该方法只需检测关键点，并从两幅图像中提取局部不变描述符(即SIFT)。
		给定关键点和特征，我们使用匹配关键点来匹配这两幅图像中的特征。
		如果返回的匹配项M为None，则没有足够的关键点匹配来创建全景图，因此我们只返回调用函数。
		否则，我们现在准备应用透视变换
		我们现在准备将两个图像拼接在一起。
		首先，我们调用cv2.warpPerspective，它需要三个参数：
		1、我们想要扭曲的图像（在这种情况下，右图像），
		2、3 x 3变换矩阵（H），
		3、最后是输出图像的形状。
		我们通过获取两个图像的宽度之和然后使用第二个图像的高度来从输出图像中导出形状。
		检查是否应该可视化KeyPoint匹配，如果应该的话，我们调用DrawMatches并将全景图和可视化的元组返回给调用方法。
		顾名思义，detectAndDescribe方法接受图像，然后检测关键点并提取局部不变描述符。在我们的实现中，我们使用高斯差分（DoG）关键点检测器和SIFT特征提取器
		'''


		# 匹配两幅图像的之间的特征
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)

		# 如果匹配为None，则没有足够的匹配的关键点用来创建全景图
		if M is None:
			return None

		# 否则（有足够的匹配的关键点），用透视图变换将图像拼接在一起）
		(matches, H, status) = M
		result = cv2.warpPerspective(imageA, H,
			(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

		# 检查关键点匹配是否应该可视化
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)

			# 返回拼接图像的元组并可视化
			return (result, vis)

		# 返回拼接的图像
		return result

	def detectAndDescribe(self, image):
		# 定义dectAndDescribe函数
		# 将图片转为灰度
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# 检查是否使用opencv3
		if self.isv3:
			# 如果使用的是opencv3使用如下的
			# 检测图片中的关键点
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)

		# 否则在opencv2.4使用如下的
		else:
			# 检测图片中的关键点
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)

			# 从图像中提取特征
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)

		# 将KeyPoint对象中的关键点转换为NumPy数组
		kps = np.float32([kp.pt for kp in kps])

		# 返回关键点和特征的元组
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# 计算原始匹配并初始化实际的匹配
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# 循环原始匹配点
		for m in rawMatches:
			# 确保距离在一定的比例范围内（即Lowe比率测试）
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# 计算一个转换矩阵至少需要4个匹配
		if len(matches) > 4:
			# 构造两组点
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# 计算两组点之间的变换矩阵
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)

			# 将带有变换矩阵的匹配点和每个匹配点的状态一起返回
			return (matches, H, status)

		# 否则（即匹配的点小于4个）不能计算变换矩阵
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# 初始化输出的可视化图像
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8") # 8位的三维0数组
		# 将两张图像放在同一张图的左右部分
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# 在匹配点中循环操作
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# 仅在关键点成功匹配时进行匹配
			if s == 1:
				# 画出匹配关系
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# 返回输出可视化之后的图像
		return vis