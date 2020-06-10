//package de.csbdresden.n2v.train;
//
//import de.csbdresden.denoiseg.train.DenoiSegConfig;
//import de.csbdresden.denoiseg.train.DenoiSegTraining;
//import net.imagej.ImageJ;
//import net.imglib2.FinalDimensions;
//import net.imglib2.img.Img;
//import net.imglib2.type.numeric.real.FloatType;
//import org.junit.Test;
//
//import java.util.Random;
//
//public class N2VTrainingTest {
//
//	@Test
//	public void testTrainingValidationBatches2D() {
//
//		ImageJ ij = new ImageJ();
//		ij.ui().setHeadless(true);
//		Img<FloatType> trainingBatches = ij.op().create().img(new FinalDimensions(32, 32, 128), new FloatType());
//		Random random = new Random();
//		trainingBatches.forEach(pix -> pix.set(random.nextFloat()));
//		Img<FloatType> validationBatches = ij.op().create().img(new FinalDimensions(32, 32, 4), new FloatType());
//		validationBatches.forEach(pix -> pix.set(random.nextFloat()));
//
//		long batchSize = trainingBatches.dimension(2);
//
////		for (int i = 0; i < 10; i++) {
//			DenoiSegTraining n2v = new DenoiSegTraining(ij.context());
//			n2v.init(new DenoiSegConfig()
//					.setTrainDimensions(2)
//					.setNumEpochs(2)
//					.setStepsPerEpoch(2)
//					.setBatchSize((int)batchSize)
//					.setBatchDimLength(32)
//					.setPatchDimLength(32));
//			n2v.input().addTrainingData(trainingBatches);
//			n2v.input().addValidationData(validationBatches);
//			n2v.train();
//			n2v.dispose();
////		}
//
//		ij.context().dispose();
//	}
//
//	@Test
//	public void testTrainingValidationBatches3D() {
//
//		ImageJ ij = new ImageJ();
//		ij.ui().setHeadless(true);
//		Img<FloatType> trainingBatches = ij.op().create().img(new FinalDimensions(32, 32, 32, 32), new FloatType());
//		Random random = new Random();
//		trainingBatches.forEach(pix -> pix.set(random.nextFloat()));
//		Img<FloatType> validationBatches = ij.op().create().img(new FinalDimensions(32, 32, 32, 2), new FloatType());
//		validationBatches.forEach(pix -> pix.set(random.nextFloat()));
//
//		long batchSize = trainingBatches.dimension(3);
//
//		DenoiSegTraining n2v = new DenoiSegTraining(ij.context());
//		n2v.init(new DenoiSegConfig()
//				.setTrainDimensions(3)
//				.setNumEpochs(1)
//				.setStepsPerEpoch(2)
//				.setBatchSize((int)batchSize)
//				.setBatchDimLength(32)
//				.setPatchDimLength(16));
//		n2v.input().addTrainingData(trainingBatches);
//		n2v.input().addValidationData(validationBatches);
//		n2v.train();
//
//		ij.context().dispose();
//	}
//}
