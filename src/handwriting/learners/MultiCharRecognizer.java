package handwriting.learners;

import handwriting.core.Drawing;
import handwriting.core.RecognizerAI;
import handwriting.core.SampleData;

import java.util.concurrent.ArrayBlockingQueue;

/**
 * Created by josephbenton on 9/28/15.
 */
public class MultiCharRecognizer implements RecognizerAI {
    MultiLayer net;
    double rate;
    int iterations;

    public MultiCharRecognizer() {
        this.net = new MultiLayer(1600, 30, 8);
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
        double[] target = new double[8];
        switch (label) {
            case "A":
                target[0] = 1;
                break;
            case "B":
                target[1] = 1;
                break;
            case "C":
                target[2] = 1;
                break;
            case "D":
                target[3] = 1;
                break;
            case "E":
                target[4] = 1;
                break;
            case "F":
                target[5] = 1;
                break;
            case "G":
                target[6] = 1;
                break;
            case "H":
                target[7] = 1;
                break;
        }
        return target;
    }

    @Override
    public String classify(Drawing d) {
        net.compute(drawing2doubles(d));
        return (net.getOutputLayer().output(0) > 0.5) ? "X" : "O";
    }

}
