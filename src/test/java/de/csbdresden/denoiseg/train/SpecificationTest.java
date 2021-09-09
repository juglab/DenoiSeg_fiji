/*-
 * #%L
 * DenoiSeg plugin
 * %%
 * Copyright (C) 2019 - 2020 Center for Systems Biology Dresden
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
package de.csbdresden.denoiseg.train;

import io.bioimage.specification.io.SpecificationReader;
import io.bioimage.specification.io.SpecificationWriter;
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
		SpecificationReader.readFromZIP(savedModel, spec);
		File tmpDir = Files.createTempDirectory("denoiseg-spec-test").toFile();
		SpecificationWriter.write(spec, tmpDir);
		String content = FileUtils.readFileToString(new File(tmpDir, SpecificationWriter.getModelFileName()));
		System.out.println(content);
	}

	public static void main(String... args) throws IOException {
		DenoiSegModelSpecification spec = new DenoiSegModelSpecification();
		File tmpDir = Files.createTempDirectory("denoiseg-spec-test").toFile();
		SpecificationWriter.write(spec, tmpDir);
		String content = FileUtils.readFileToString(new File(tmpDir, SpecificationWriter.getModelFileName()));
		System.out.println(content);
	}

}
