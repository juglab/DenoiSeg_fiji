package de.csbdresden.denoiseg.interactive;

import de.csbdresden.denoiseg.command.DenoiSegTrainPredictCommand;
import de.csbdresden.denoiseg.train.DenoiSegConfig;
import de.csbdresden.denoiseg.train.DenoiSegTraining;
import net.imagej.ImageJ;
import net.imagej.modelzoo.DefaultModelZooArchive;
import net.imglib2.FinalDimensions;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.FloatType;
import org.junit.Test;
import org.scijava.command.CommandModule;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.ExecutionException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class OneClickTest {

	@Test
	public void testTrainingAndPrediction() throws ExecutionException, InterruptedException, IOException {

		ImageJ ij = new ImageJ();
		ij.ui().setHeadless(true);
		Img<FloatType> trainingBatches = ij.op().create().img(new FinalDimensions(32, 32, 32), new FloatType());
		Random random = new Random();
		trainingBatches.forEach(pix -> pix.set(random.nextFloat()));
		Img<FloatType> predictionBatches = ij.op().create().img(new FinalDimensions(32, 32, 4), new FloatType());
		predictionBatches.forEach(pix -> pix.set(random.nextFloat()));

		DenoiSegTraining training = new DenoiSegTraining(ij.context());
		training.init(new DenoiSegConfig()
			.setBatchSize(10)
			.setNumEpochs(2)
			.setStepsPerEpoch(2)
			.setPatchShape(32)
			.setNeighborhoodRadius(5));
		for (int i = 0; i < 20; i++) {
			Img<FloatType> raw = ij.op().create().img(new FinalDimensions(32, 32), new FloatType());
			Img<IntType> labeling = null;
			if(i < 5) labeling = ij.op().create().img(new FinalDimensions(32, 32), new IntType());
			training.input().addTrainingData(raw, labeling);
			training.input().addValidationData(raw, labeling);
		}
		training.train();
		File modelFile = training.output().exportLatestTrainedModel();
		assertNotNull(modelFile);
		Object model = ij.io().open(modelFile.getAbsolutePath());
		assertNotNull(model);
		assertEquals(DefaultModelZooArchive.class, model.getClass());
	}
}
