package de.csbdresden.denoiseg.command;

import net.imagej.ImageJ;
import net.imglib2.RandomAccessibleInterval;
import org.junit.Test;
import org.scijava.command.CommandModule;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;

import static org.junit.Assert.assertNotNull;

public class TestPredictCommand {
	@Test
	public void predictViaCommand() throws ExecutionException, InterruptedException, IOException {

		final ImageJ ij = new ImageJ();

		String modelPath = getClass().getResource("/denoiseg-dummy.bioimage.io.zip").getPath();

		RandomAccessibleInterval _input = ij.op().create().img(new int[]{20, 20});

		CommandModule plugin = ij.command().run( DenoiSegPredictCommand.class, false
				,"input", _input, "modelFile", new File(modelPath)
		).get();

		assertNotNull( plugin.getOutput( "output" ) );

		ij.dispose();

	}
}
