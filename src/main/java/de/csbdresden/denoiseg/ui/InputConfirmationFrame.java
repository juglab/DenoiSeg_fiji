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
package de.csbdresden.denoiseg.ui;

import de.csbdresden.denoiseg.train.XYPairs;
import jdk.nashorn.internal.scripts.JD;
import net.imagej.display.ColorTables;
import net.imagej.display.SourceOptimizedCompositeXYProjector;
import net.imagej.ops.OpService;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.stats.ComputeMinMax;
import net.imglib2.converter.RealLUTConverter;
import net.imglib2.display.projector.composite.CompositeXYProjector;
import net.imglib2.display.screenimage.awt.ARGBScreenImage;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.cell.AbstractCellImg;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import net.miginfocom.swing.MigLayout;
import org.scijava.Context;
import org.scijava.plugin.Parameter;

import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import java.awt.Component;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collections;

public class InputConfirmationFrame<T extends RealType<T> & NativeType<T>> extends JPanel {

	@Parameter
	private OpService opService;

	private int numImagesDisplayed = 5;
	private int previewDim = 200;

	public InputConfirmationFrame(Context context, XYPairs<T> imagePairs) {
		super(new MigLayout());
		context.inject(this);
		ImageIcon inputIcon = new ImageIcon();
		ImageIcon outputIcon = new ImageIcon();
		add(new JLabel("Does the upper row match the lower row?"));
		add(createPreviewPanel(inputIcon, outputIcon, imagePairs), "newline, push, grow, span");
	}

	private Component createPreviewPanel(ImageIcon inputIcon, ImageIcon outputIcon, XYPairs<T> imagePairs) {
		JPanel panel = new JPanel(new MigLayout());
		XYPairs<T> list = new XYPairs<>();
		list.addAll(imagePairs);
		Collections.shuffle(list);
		inputIcon.setImage(toBufferedImage(toImageRow(list, true)));
		outputIcon.setImage(toBufferedImage(toImageRow(list, false)));
		JLabel label1 = new JLabel(inputIcon);
		JLabel label2 = new JLabel(outputIcon);
		label1.setBorder(BorderFactory.createEmptyBorder(10, 10, 5, 10));
		label2.setBorder(BorderFactory.createEmptyBorder(5, 10, 10, 10));
		panel.add(label1, "height 100:100:100, width 500:500:500");
		panel.add(label2, "newline, height 100:100:100, width 500:500:500");
		return panel;
	}

	private RandomAccessibleInterval<T> toImageRow(XYPairs<T> list, boolean useFirst) {

		RandomAccessibleInterval<T> firstImage = list.get(0).getA();
		while(firstImage.numDimensions() > 2) {
			firstImage = Views.hyperSlice(firstImage, 2, 0);
		}
		long[] tileDims = new long[firstImage.numDimensions()];
		firstImage.dimensions(tileDims);

		long[] dims = new long[tileDims.length];
		dims[0] = tileDims[0] * numImagesDisplayed;
		for (int i = 1; i < dims.length; i++) {
			dims[i] = tileDims[i];
		}
		Img<T> res = new ArrayImgFactory<>(list.get(0).getA().randomAccess().get()).create(dims);
		long[] minSize = new long[dims.length*2];
		for (int i = 0; i < dims.length; i++) {
			minSize[i] = 0;
			minSize[i+dims.length] = list.get(0).getA().dimension(i);
		}
		for (int i = 0; i < numImagesDisplayed; i++) {
			RandomAccessibleInterval<T> tile = useFirst ? list.get(i).getA() : list.get(i).getB();
			while(tile.numDimensions() > 2) {
				tile = Views.hyperSlice(tile, 2, 0);
			}
			minSize[0] = i * tileDims[0];
			FinalInterval interval = Intervals.createMinSize(minSize);
			LoopBuilder.setImages(Views.zeroMin(Views.interval(res, interval)), tile)
					.multiThreaded().forEachPixel(net.imglib2.type.Type::set);
		}
		return res;
	}

	private BufferedImage toBufferedImage(RandomAccessibleInterval<T> img) {
		while(img.numDimensions() > 2) {
			img = Views.hyperSlice(img, 2, 0);
		}
		img = opService.transform().scaleView(img, new double[]{(double)previewDim*5/(double)img.dimension(0), (double)previewDim/(double)img.dimension(1)}, new NLinearInterpolatorFactory<>());
		ARGBScreenImage screenImage = new ARGBScreenImage((int)img.dimension(0), (int)img.dimension(1));
		T min = img.randomAccess().get().copy();
		T max = img.randomAccess().get().copy();
		ComputeMinMax<T> minMax = new ComputeMinMax<>(Views.iterable(img), min, max);
		minMax.process();
		RealLUTConverter<? extends RealType<?>> converter = new RealLUTConverter(min.getRealDouble(),
				max.getRealDouble(), ColorTables.GRAYS);
		ArrayList<RealLUTConverter<? extends RealType<?>>> converters = new ArrayList<>();
		converters.add(converter);
		CompositeXYProjector projector;
		if (AbstractCellImg.class.isAssignableFrom(img.getClass())) {
			projector =
					new SourceOptimizedCompositeXYProjector(img,
							screenImage, converters, -1);
		}
		else {
			projector =
					new CompositeXYProjector(img, screenImage,
							converters, -1);
		}
		projector.setComposite(false);
		projector.map();
		return screenImage.image();
	}

	public boolean confirm() {
		String[] options = new String[]{"Yes, continue training", "No, stop training"};
		int res = JOptionPane.showOptionDialog(null, this, "Confirm input data", JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE, null, options, options[0]);
		System.out.println(res);
		return res == 0;
	}
}
