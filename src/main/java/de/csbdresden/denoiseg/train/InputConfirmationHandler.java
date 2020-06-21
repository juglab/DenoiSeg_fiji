package de.csbdresden.denoiseg.train;

import de.csbdresden.denoiseg.ui.InputConfirmationFrame;
import org.scijava.Context;
import org.scijava.plugin.Parameter;
import org.scijava.ui.UIService;

import java.io.File;

public class InputConfirmationHandler {

	@Parameter
	private UIService uiService;

	@Parameter
	private Context context;

	private final InputHandler inputHandler;

	public InputConfirmationHandler(Context context, InputHandler inputHandler) {
		context.inject(this);
		this.inputHandler = inputHandler;
	}

	public boolean confirmTrainingData() {
		InputConfirmationFrame frame = new InputConfirmationFrame(context, inputHandler.getLabeledTrainingPairs());
		return frame.confirm();
	}

	public boolean confirmValidationData() {
		return false;
	}
}
