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
package de.csbdresden.denoiseg.command;

import de.csbdresden.denoiseg.predict.DenoiSegPrediction;
import de.csbdresden.denoiseg.train.TrainUtils;
import io.scif.MissingLibraryException;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.modelzoo.consumer.commands.SingleImagePredictionCommand;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.converter.RealFloatConverter;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.command.CommandModule;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

@Plugin( type = SingleImagePredictionCommand.class, name = "denoiseg", menuPath = "Plugins>CSBDeep>DenoiSeg>DenoiSeg predict" )
public class DenoiSegPredictCommand<T extends RealType<T>> implements SingleImagePredictionCommand {

	@Parameter(label = "Trained model file (.zip)")
	private File modelFile;

	@Parameter
	private RandomAccessibleInterval< T > input;

	@Parameter(label = "Axes of prediction input (subset of XYB, B = batch)", description = "<html>You can predict one dimension independently per position.<br>Use B ( = batch) for this dimension.")
	private String axes = "XY";

	@Parameter(label = "Batch size", required = false, description = "<html>The batch size will only be used if a batch axis exists.<br>It can improve performance to process multiple batches at once (batch size > 1)")
	private int batchSize = 10;

	@Parameter(label = "Number of tiles (1 = no tiling)", required = false, description = "<html>Increasing the tiling can help if the memory is insufficient to deal with the whole image at once.<br>Too many tiles decrease performance because an overlap has to be computed.")
	private int numTiles = 1;

	@Parameter( type = ItemIO.OUTPUT )
	private Dataset output;

	@Parameter
	private Context context;

//	@Parameter
//	private DisplayService displayService;

	@Parameter
	private DatasetService datasetService;

//	@Parameter
//	private ImageDisplayService imageDisplayService;

	@Override
	public void run() {
		//TODO make transferring LUTs work..
		//TODO the following code works for IJ2, but not for LUTs set via IJ1
//		List<Display<?>> displays = displayService.getDisplays(prediction);
//		List<ColorTable> colorTables = new ArrayList<>();
//		if(displays.size() > 0) {
//			ImageDisplay display = (ImageDisplay) displays.get(0);
//			display.update();
//			DatasetView view = imageDisplayService.getActiveDatasetView(display);
//			colorTables = view.getColorTables();
//		}
		DenoiSegPrediction prediction = new DenoiSegPrediction(context);
		try {
			prediction.setTrainedModel(modelFile.getAbsolutePath());
		} catch (IOException e) {
			e.printStackTrace();
		}
		prediction.setNumberOfTiles(numTiles);
		prediction.setBatchSize(batchSize);
//		output = Converters.convert(_output, new FloatRealConverter<>(), input.randomAccess().get());
//		prediction.predict(converted);
		RandomAccessibleInterval<FloatType> rai = null;
		try {
			rai = prediction.predict(input, axes);
		} catch (Exception e) {
			e.printStackTrace();
		}
		if(rai == null) return;
		output = datasetService.create(rai);
		output.setRGBMerged(false);

//		output = datasetService.create(_output);
//		output.initializeColorTables(colorTables.size());
//		for (int i = 0; i < colorTables.size(); i++) {
//			output.setColorTable(colorTables.get(i), i);
//		}
	}

	public static void main( final String... args ) throws Exception {

		final ImageJ ij = new ImageJ();

		ij.launch( args );

//		ij.log().setLevel(LogLevel.TRACE);

		File modelFile = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/mouse/Mouse_n10/results/n2v-latest-6442216215584512595/denoiseg-mouse.bioimage.io.zip");

		final File predictionInput = new File( "/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/X_train/img_3.tif" );

		if ( predictionInput.exists() ) {
			RandomAccessibleInterval _input = ( RandomAccessibleInterval ) ij.io().open( predictionInput.getAbsolutePath() );
			RandomAccessibleInterval _inputConverted = ij.op().convert().float32( Views.iterable( _input ) );
//			_inputConverted = Views.interval(_inputConverted, new FinalInterval(1024, 1024  ));

			RandomAccessibleInterval prediction = ij.op().copy().rai( _inputConverted );

			ij.ui().show(prediction);

			CommandModule plugin = ij.command().run( DenoiSegPredictCommand.class, true
//					,"input", prediction, "modelFile", modelFile
			).get();
			ij.ui().show( plugin.getOutput( "output" ) );
		} else
			System.out.println( "Cannot find training image " + predictionInput.getAbsolutePath() );

	}
}
