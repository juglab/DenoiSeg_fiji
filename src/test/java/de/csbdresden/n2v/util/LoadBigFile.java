//package de.csbdresden.n2v.util;
//
//import de.csbdresden.denoiseg.train.DenoiSegConfig;
//import de.csbdresden.denoiseg.train.DenoiSegTraining;
//import io.scif.FormatException;
//import io.scif.Reader;
//import io.scif.SCIFIO;
//import io.scif.config.SCIFIOConfig;
//import io.scif.img.ImgOpener;
//import io.scif.img.SCIFIOImgPlus;
//import net.imagej.ImageJ;
//import net.imglib2.RandomAccessibleInterval;
//import net.imglib2.img.Img;
//import net.imglib2.type.numeric.real.FloatType;
//import org.scijava.io.location.FileLocation;
//
//import java.io.IOException;
//import java.util.List;
//
//public class LoadBigFile {
//
//	public static void main(String...args) throws IOException, FormatException {
//		ImageJ ij = new ImageJ();
//
//		ij.launch();
//
//		final String imageFile = "/home/random/Development/imagej/project/3DAnalysisFIBSegmentation/G64-2-1_HighGlucose 4x4x4nm_cell_1.tif";
//
//		final SCIFIOConfig config = new SCIFIOConfig();
//		config.imgOpenerSetImgModes( SCIFIOConfig.ImgMode.PLANAR );
//
//		final SCIFIO scifio = new SCIFIO();
//		final Reader reader = scifio.initializer().initializeReader( new FileLocation(imageFile) );
//
//		final List<SCIFIOImgPlus< ? >> imgs = new ImgOpener( ij.context() ).openImgs( reader, config );
//
//		DenoiSegTraining n2v = new DenoiSegTraining(ij.context());
//		n2v.init(new DenoiSegConfig()
//				.setTrainDimensions(2)
//				.setNumEpochs(100)
//				.setStepsPerEpoch(300)
//				.setBatchSize(32)
//				.setBatchDimLength(180)
//				.setPatchDimLength(60));
//		imgs.forEach(img -> {
//			Img img1 = img.getImg();
//			RandomAccessibleInterval<FloatType> converted = ij.op().convert().float32(img1);
//			n2v.input().addTrainingAndValidationData(converted, 0.1);
//		});
//		n2v.train();
//	}
//
//}
