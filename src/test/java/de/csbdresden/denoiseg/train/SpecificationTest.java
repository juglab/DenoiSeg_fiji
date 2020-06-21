package de.csbdresden.denoiseg.train;

import net.imagej.ImageJ;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.FloatType;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class SpecificationTest {

//	@Test
	public void testTrainingSpec() throws IOException {
		ImageJ ij = new ImageJ();
		DenoiSegTraining training = new DenoiSegTraining(ij.context());
		training.init(new DenoiSegConfig()
				.setNumEpochs(1)
				.setStepsPerEpoch(1)
				.setBatchSize(2)
				.setPatchShape(2)
				.setNeighborhoodRadius(1));
		training.input().addTrainingData(new ArrayImgFactory(new FloatType()).create(10, 10), new ArrayImgFactory(new IntType()).create(10, 10));
		training.input().addTrainingData(new ArrayImgFactory(new FloatType()).create(10, 10), new ArrayImgFactory(new IntType()).create(10, 10));
		training.input().addValidationData(new ArrayImgFactory(new FloatType()).create(10, 10), new ArrayImgFactory(new IntType()).create(10, 10));
		training.input().addValidationData(new ArrayImgFactory(new FloatType()).create(10, 10), new ArrayImgFactory(new IntType()).create(10, 10));
		training.train();
		File savedModel = training.output().exportLatestTrainedModel();
		DenoiSegModelSpecification spec = new DenoiSegModelSpecification();
		spec.readFromZIP(savedModel);
		File tmpDir = Files.createTempDirectory("denoiseg-spec-test").toFile();
		spec.write(tmpDir);
		String content = FileUtils.readFileToString(new File(tmpDir, spec.getModelFileName()));
		System.out.println(content);
	}

	public static void main(String... args) throws IOException {
		DenoiSegModelSpecification spec = new DenoiSegModelSpecification();
		File tmpDir = Files.createTempDirectory("denoiseg-spec-test").toFile();
		spec.write(tmpDir);
		String content = FileUtils.readFileToString(new File(tmpDir, spec.getModelFileName()));
		System.out.println(content);
	}

}
