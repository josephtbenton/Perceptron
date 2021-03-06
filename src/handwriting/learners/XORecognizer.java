package handwriting.learners;

import handwriting.core.Drawing;
import handwriting.core.RecognizerAI;
import handwriting.core.SampleData;

import java.util.concurrent.ArrayBlockingQueue;

/**
 * Created by josephbenton on 9/28/15.
 */
public class XORecognizer implements RecognizerAI {
    MultiLayer net;
    double rate;
    int iterations;

    public XORecognizer() {
        this.net = new MultiLayer(1600, 30, 1);
        rate = 0.05;
        this.iterations = 2000;
    }

    @Override
    public void train(SampleData data, ArrayBlockingQueue<Double> progress) throws InterruptedException {
        for (int j = 0; j < iterations; j++) {
            progress.put((double)j / (double)iterations);
            for (String label: data.allLabels()) {
                for (int i = 0; i < data.numDrawingsFor(label); ++i) {
                    net.train(drawing2doubles(data.getDrawing(label, i)), label2doubles(label), rate);
                }
                net.updateWeights();
            }
        }
    }

    private double[] drawing2doubles(Drawing drawing) {
        int width = drawing.getWidth();
        int height = drawing.getHeight();
        int cur = 0;
        double[] inputs = new double[width * height];
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                inputs[cur] = drawing.isSet(x, y) ? 1.0 : 0.0;
                cur++;
            }
        }
        return inputs;
    }

    private double[] label2doubles(String label) {
        double[] target = new double[1];
        if (label.equals("X")) {
            target[0] = 1.0;
        } else if (label.equals("O")) {
            target[0] = 0.0;
        }
        return target;
    }

    @Override
    public String classify(Drawing d) {
        net.compute(drawing2doubles(d));
        return (net.getOutputLayer().output(0) > 0.5) ? "X" : "O";
    }

}
