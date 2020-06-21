package de.csbdresden.denoiseg.train;

import net.imagej.ops.OpService;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.display.Display;
import org.scijava.display.DisplayService;
import org.scijava.plugin.Parameter;
import org.scijava.ui.UIService;

import java.util.Arrays;

public class PreviewHandler {

	@Parameter
	private UIService uiService;

	private final int trainDimensions;
	private RandomAccessibleInterval<FloatType> trainingImage;
	private RandomAccessibleInterval<FloatType> validationImage;

	private RandomAccessibleInterval<FloatType> singleValidationInputImage;
	private RandomAccessibleInterval<FloatType> singleValidationOutputImage;

	public PreviewHandler(Context context, int trainDimensions) {
		context.inject(this);
		this.trainDimensions = trainDimensions;
	}

	public void updateValidationPreview(RandomAccessibleInterval<FloatType> in, RandomAccessibleInterval<FloatType> out) {
		if (Thread.interrupted()) return;
		singleValidationInputImage = Views.hyperSlice(in, in.numDimensions()-2, 0);
		singleValidationOutputImage = Views.hyperSlice(out, out.numDimensions()-2, 0);
		long[] dims = new long[in.numDimensions()-1];
		int channelCount = 5;
		dims[0] = in.dimension(0)* channelCount;
		for (int i = 1; i < dims.length; i++) {
			dims[i] = in.dimension(i);
		}
		if(validationImage == null) {
			validationImage = new ArrayImgFactory<>(new FloatType()).create(dims);
		}
		long[] minSize = new long[dims.length*2];
		for (int i = 0; i < dims.length; i++) {
			minSize[i] = 0;
			minSize[i+dims.length] = in.dimension(i);
		}
		for (int i = 0; i < channelCount; i++) {
			minSize[0] = i*in.dimension(0);
			FinalInterval interval = Intervals.createMinSize(minSize);
			RandomAccessibleInterval<FloatType> source;
			if(i == 0) {
				source = Views.hyperSlice(in, dims.length, 0);
			} else {
				source = Views.hyperSlice(out, dims.length, i-1);
			}
			LoopBuilder.setImages(Views.zeroMin(Views.interval(validationImage, interval)), source)
					.forEachPixel(FloatType::set);
		}

//		else LoopBuilder.setImages(splitImage, out).forEachPixel((o, i) -> o.set(i));
//		else opService.copy().rai(splitImage, (RandomAccessibleInterval)out);
//		if(trainDimensions == 2) updateSplitImage2D(in);
//		if(trainDimensions == 3) updateSplitImage3D(in);
		Display<?> display = uiService.context().service(DisplayService.class).getDisplay("training preview");
		if(display == null) uiService.show("training preview", validationImage);
		else display.update();
	}

	RandomAccessibleInterval<FloatType> getExampleInput() {
		return singleValidationInputImage;
	}

	RandomAccessibleInterval<FloatType> getExampleOutput() {
		return singleValidationOutputImage;
	}
}
