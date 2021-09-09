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
package de.csbdresden.denoiseg;

import de.csbdresden.denoiseg.train.DenoiSegConfig;
import de.csbdresden.denoiseg.train.DenoiSegTraining;
import net.imagej.ImageJ;
import net.imagej.modelzoo.DefaultModelZooArchive;
import net.imglib2.FinalDimensions;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.FloatType;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class TrainingTest {

	@Test
	public void testTrainingAndPrediction() throws IOException {

		ImageJ ij = new ImageJ();
		ij.ui().setHeadless(true);
		Random random = new Random();

		DenoiSegTraining training = new DenoiSegTraining(ij.context());
		training.init(new DenoiSegConfig()
			.setBatchSize(64)
			.setNumEpochs(2)
			.setStepsPerEpoch(2)
			.setPatchShape(32)
			.setNeighborhoodRadius(5));
		for (int i = 0; i < 20; i++) {
			Img<FloatType> raw = ij.op().create().img(new FinalDimensions(322, 322), new FloatType());
			raw.forEach(pix -> pix.set(random.nextFloat()));
			Img<IntType> labeling = null;
			if(i < 5) {
				labeling = ij.op().create().img(new FinalDimensions(322, 322), new IntType());
				labeling.forEach(pix -> pix.add(new IntType(random.nextFloat() > 0.5 ? 1 : 0 )));
			}
			training.input().addTrainingData(raw, labeling);
			training.input().addValidationData(raw, labeling);
		}
		training.train();
		File modelFile = training.output().exportLatestTrainedModel();
		assertNotNull(modelFile);
		Object model = ij.io().open(modelFile.getAbsolutePath());
		assertNotNull(model);
		assertEquals(DefaultModelZooArchive.class, model.getClass());
		ij.context().dispose();
	}
}
