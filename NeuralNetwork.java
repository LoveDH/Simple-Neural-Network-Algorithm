import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.io.*;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.util.Random;
//*********************************************************************

class ImagePanel extends JPanel {
	Image image;

	public ImagePanel(Image image) {
		this.image = image;
	}

	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		g.drawImage(image, 0, 0, this);
	}

	// drawing Neural Network
	public void paint(Graphics g) {

		int[] x_point = { 110, 320, 530, 740 };
		int[][] y_point = new int[4][];
		y_point[0] = new int[4];
		y_point[1] = new int[9];
		y_point[2] = new int[9];
		y_point[3] = new int[3];

		for (int i = 0; i < 4; i++) {
			y_point[0][i] = 160 + i * 80;
		}
		for (int i = 0; i < 9; i++) {
			y_point[1][i] = 80 + i * 50;
		}
		for (int i = 0; i < 9; i++) {
			y_point[2][i] = 80 + i * 50;
		}
		for (int i = 0; i < 3; i++) {
			y_point[3][i] = 180 + i * 100;
		}

		super.paint(g);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 9; j++) {
				g.drawLine(x_point[0] + 60, y_point[0][i] + 10, x_point[1], y_point[1][j] + 10);
			}
		}
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				g.drawLine(x_point[1] + 60, y_point[1][i] + 10, x_point[2], y_point[2][j] + 10);
			}
		}
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 3; j++) {
				g.drawLine(x_point[2] + 60, y_point[2][i] + 10, x_point[3], y_point[3][j] + 10);
			}
		}
	}
}

// *********************************************************************

public class NeuralNetwork {
	public static void main(String args[]) throws FileNotFoundException {

		String filepath = "./data/iris_train_ns.txt"; // data path
		// if you want to use sorted data, insert file 'iris_train.txt'
		Operations op = new Operations(filepath);
		op.getInterface();
	}
}

// *********************************************************************

class Operations {

	// defining weight and history arrays
	double[][] input = new double[4][150];
	double[][] weight_IH = new double[4][9];
	double[][] weight_HH = new double[9][9];
	double[][] weight_HO = new double[9][3];
	ArrayList<Double>[][] WIH_HIST = new ArrayList[4][9];
	ArrayList<Double>[][] WHH_HIST = new ArrayList[9][9];
	ArrayList<Double>[][] WHO_HIST = new ArrayList[9][3];
	double[] bias_hidden1 = new double[9];
	double[] bias_hidden2 = new double[9];
	double[] bias_output = new double[3];
	ArrayList<Double>[] BH1_HIST = new ArrayList[9];
	ArrayList<Double>[] BH2_HIST = new ArrayList[9];
	ArrayList<Double>[] BO_HIST = new ArrayList[3];
	int iris_type;
	double MSE;
	double LearningRate;

	// nodes list
	ArrayList<Node> InList = new ArrayList<Node>();
	ArrayList<Node> HidList1 = new ArrayList<Node>();
	ArrayList<Node> HidList2 = new ArrayList<Node>();
	ArrayList<Node> OutList = new ArrayList<Node>();

	FileReader fr = null;
	BufferedReader br = null;

	public Operations(String filename) throws FileNotFoundException {

		fr = new FileReader(filename);
		br = new BufferedReader(fr);
		Random oRandom = new Random();

		// inserting nodes to array list
		for (int i = 0; i < 4; i++)
			InList.add(new Node());
		for (int i = 0; i < 9; i++)
			HidList1.add(new Node());
		for (int i = 0; i < 9; i++)
			HidList2.add(new Node());
		for (int i = 0; i < 3; i++)
			OutList.add(new Node());

		// initializing weights with gaussian distribution
		for (int i = 0; i < InList.size(); i++) {
			for (int j = 0; j < HidList1.size(); j++) {
				weight_IH[i][j] = oRandom.nextGaussian() * 0.05;
				WIH_HIST[i][j] = new ArrayList<Double>();
				WIH_HIST[i][j].add(weight_IH[i][j]);
			}
		}

		for (int i = 0; i < HidList1.size(); i++) {
			for (int j = 0; j < HidList2.size(); j++) {
				weight_HH[i][j] = oRandom.nextGaussian() * 0.05;
				WHH_HIST[i][j] = new ArrayList<Double>();
				WHH_HIST[i][j].add(weight_HH[i][j]);
			}
		}

		for (int i = 0; i < HidList2.size(); i++) {
			for (int j = 0; j < OutList.size(); j++) {
				weight_HO[i][j] = oRandom.nextGaussian() * 0.05;
				WHO_HIST[i][j] = new ArrayList<Double>();
				WHO_HIST[i][j].add(weight_HO[i][j]);
			}
		}

		for (int i = 0; i < HidList1.size(); i++) {
			bias_hidden1[i] = 0;

			BH1_HIST[i] = new ArrayList<Double>();
			BH1_HIST[i].add(bias_hidden1[i]);
		}

		for (int i = 0; i < HidList2.size(); i++) {
			bias_hidden2[i] = 0;

			BH2_HIST[i] = new ArrayList<Double>();
			BH2_HIST[i].add(bias_hidden2[i]);
		}

		for (int i = 0; i < OutList.size(); i++) {
			bias_output[i] = 0;

			BO_HIST[i] = new ArrayList<Double>();
			BO_HIST[i].add(bias_output[i]);
		}
	}

	void getInterface() {
		// background image
		String imageFile = "./data/background.jpg";
		Image image = Toolkit.getDefaultToolkit().getImage(imageFile);

		JFrame tstFrame = new JFrame("Multi Layer Neural Network");

		tstFrame.addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {
				System.exit(0);
			}
		});
		Container tstPane = tstFrame.getContentPane();
		ImagePanel imagePanel = new ImagePanel(image);
		SpringLayout layout = new SpringLayout();
		imagePanel.setLayout(layout);
		tstPane.add(imagePanel, BorderLayout.CENTER);

		// ********* input nodes
		final JTextField input[] = new JTextField[4];

		for (int i = 0; i < InList.size(); i++) {
			input[i] = new JTextField("", 5);
			input[i].setEditable(false);
			imagePanel.add(input[i]);
		}

		// ********* hidden1 nodes
		final JTextField hidden1[] = new JTextField[9];

		for (int i = 0; i < HidList1.size(); i++) {
			hidden1[i] = new JTextField("", 5);
			hidden1[i].setEditable(false);
			imagePanel.add(hidden1[i]);
		}

		// ********* hidden2 nodes
		final JTextField hidden2[] = new JTextField[9];

		for (int i = 0; i < HidList2.size(); i++) {
			hidden2[i] = new JTextField("", 5);
			hidden2[i].setEditable(false);
			imagePanel.add(hidden2[i]);
		}

		// ********* output nodes
		final JTextField output[] = new JTextField[3];

		for (int i = 0; i < OutList.size(); i++) {
			output[i] = new JTextField("", 5);
			output[i].setEditable(false);
			imagePanel.add(output[i]);
		}

		// ********* Label nodes
		final JTextField label[] = new JTextField[3];

		for (int i = 0; i < 3; i++) {
			label[i] = new JTextField("", 5);
			label[i].setEditable(false);
			imagePanel.add(label[i]);
		}

		// *******weight history viewer
		JTextArea weightRecord = new JTextArea("", 15, 10);
		weightRecord.setEditable(false);
		JScrollPane scrollRecord = new JScrollPane(weightRecord);

		imagePanel.add(scrollRecord);

		// ********* learning rate box
		JLabel lr = new JLabel("Learning Rate:", JLabel.LEFT);
		imagePanel.add(lr);
		final JTextField rateText = new JTextField("0.1", 5);
		imagePanel.add(rateText);

		// ********* error box
		final JTextField errorText = new JTextField("MeanSquaredError", 10);
		errorText.setEditable(false);
		imagePanel.add(errorText);

		// ********* RUN button
		JButton tstButton1 = new JButton("Run");
		imagePanel.add(tstButton1);
		tstButton1.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				String line;
				try {
					line = br.readLine();
					StringTokenizer st = new StringTokenizer(line);
					int lb1, lb2, lb3, type;
					double SL, SW, PL, PW;
					SL = Double.parseDouble(st.nextToken());
					SW = Double.parseDouble(st.nextToken());
					PL = Double.parseDouble(st.nextToken());
					PW = Double.parseDouble(st.nextToken());
					st.nextToken();
					lb1 = Integer.parseInt(st.nextToken());
					lb2 = Integer.parseInt(st.nextToken());
					lb3 = Integer.parseInt(st.nextToken());
					type = Integer.parseInt(st.nextToken());
					iris_type = type;
					LearningRate = Double.parseDouble(rateText.getText());

					double[] inputs = { SL, SW, PL, PW };
					double[] labels = { lb1, lb2, lb3 };

					for (int i = 0; i < InList.size(); i++) {
						input[i].setText(Double.toString(inputs[i]));
					}
					for (int i = 0; i < 3; i++) {
						label[i].setText(Double.toString(labels[i]));
					}

					// traing here
					performTrain(inputs, labels);

					for (int i = 0; i < HidList1.size(); i++) {
						hidden1[i].setText(String.format("%.6f", HidList1.get(i).y));
					}

					for (int i = 0; i < HidList2.size(); i++) {
						hidden2[i].setText(String.format("%.6f", HidList2.get(i).y));
					}

					for (int i = 0; i < OutList.size(); i++) {
						output[i].setText(String.format("%.4f", OutList.get(i).y));
					}

					errorText.setText(String.format("%.4f", MSE));
					String w = "";

					//********* insert HIST array which you want to check ex) WHO_HIST[1][2]
					for (double i : WHH_HIST[0][0]) { 
						w += Double.toString(i) + "\n";
					}
					weightRecord.setText(w);

				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		});

		// nodes location
		layout.putConstraint(SpringLayout.WEST, input[0], 110, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, input[0], 160, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, input[1], 110, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, input[1], 240, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, input[2], 110, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, input[2], 320, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, input[3], 110, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, input[3], 400, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden1[0], 320, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden1[0], 80, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden1[1], 320, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden1[1], 130, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden1[2], 320, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden1[2], 180, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden1[3], 320, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden1[3], 230, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden1[4], 320, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden1[4], 280, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden1[5], 320, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden1[5], 330, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden1[6], 320, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden1[6], 380, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden1[7], 320, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden1[7], 430, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden1[8], 320, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden1[8], 480, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden2[0], 530, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden2[0], 80, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden2[1], 530, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden2[1], 130, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden2[2], 530, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden2[2], 180, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden2[3], 530, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden2[3], 230, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden2[4], 530, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden2[4], 280, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden2[5], 530, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden2[5], 330, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden2[6], 530, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden2[6], 380, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden2[7], 530, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden2[7], 430, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, hidden2[8], 530, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, hidden2[8], 480, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, output[0], 740, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, output[0], 180, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, output[1], 740, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, output[1], 280, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, output[2], 740, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, output[2], 380, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, label[0], 840, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, label[0], 180, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, label[1], 840, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, label[1], 280, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, label[2], 840, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, label[2], 380, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, rateText, 90, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, rateText, 0, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, errorText, 840, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, errorText, 480, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, tstButton1, 940, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, tstButton1, 70, SpringLayout.NORTH, imagePanel);

		layout.putConstraint(SpringLayout.WEST, scrollRecord, 940, SpringLayout.WEST, imagePanel);
		layout.putConstraint(SpringLayout.NORTH, scrollRecord, 150, SpringLayout.NORTH, imagePanel);

		tstFrame.setSize(new Dimension(1200, 600));
		tstFrame.setLocationRelativeTo(null);
		tstFrame.setVisible(true);

	}

	void performTrain(double[] inputs, double[] labels) {
		for (int i = 0; i < InList.size(); i++) {
			InList.get(i).y = inputs[i];
		}
		FeedFoward();
		BackPropagation();
	}

	void FeedFoward() {
		Sum(InList, HidList1, weight_IH, bias_hidden1);
		SigmHidden(HidList1);
		Sum(HidList1, HidList2, weight_HH, bias_hidden2);
		SigmHidden(HidList2);
		Sum(HidList2, OutList, weight_HO, bias_output);
		softmaxOutput();
	}

	void BackPropagation() {
		outnodeval();
		get_error();
		MSE();
		get_delta();
		update_weight();
	}

	void Sum(ArrayList<Node> from, ArrayList<Node> to, double[][] weight, double[] bias) {
		for (int i = 0; i < to.size(); i++) {
			double sum = 0;
			for (int j = 0; j < from.size(); j++) {
				sum += weight[j][i] * from.get(j).y;
			}
			sum += bias[i];
			to.get(i).x = sum;
		}
	}

	void softmaxOutput() {
		double sum = 0;
		for (int i = 0; i < OutList.size(); i++)
			sum += Math.exp(OutList.get(i).x);
		for (int i = 0; i < OutList.size(); i++)
			OutList.get(i).y = Math.exp(OutList.get(i).x) / sum;
	}

	void SigmHidden(ArrayList<Node> layer) {
		for (int i = 0; i < layer.size(); i++) {
			double temsig = 1 / (1 + Math.exp(-layer.get(i).x));
			layer.get(i).y = temsig;
		}
	}

	void outnodeval() {
		if (iris_type == 1) {
			OutList.get(0).T = 1;
			OutList.get(1).T = 0;
			OutList.get(2).T = 0;
		} else if (iris_type == 2) {
			OutList.get(0).T = 0;
			OutList.get(1).T = 1;
			OutList.get(2).T = 0;
		} else {
			OutList.get(0).T = 0;
			OutList.get(1).T = 0;
			OutList.get(2).T = 1;
		}
	}

	void get_error() {
		double minus_val;
		for (int i = 0; i < OutList.size(); i++) {
			minus_val = OutList.get(i).y - OutList.get(i).T;
			OutList.get(i).error = minus_val;
		}
	}

	void MSE() {
		MSE = 0;
		for (int i = 0; i < OutList.size(); i++) {
			MSE += Math.pow(OutList.get(i).error, 2);
		}
		MSE /= 3.0;
	}

	void get_delta() {
		for (int i = 0; i < OutList.size(); i++) {
			OutList.get(i).delta = ((OutList.get(i).y) * (1 - OutList.get(i).y))
					* (OutList.get(i).T - OutList.get(i).y);
		}
		for (int i = 0; i < HidList2.size(); i++) {
			HidList2.get(i).delta = 0;
			for (int j = 0; j < OutList.size(); j++) {
				HidList2.get(i).delta += HidList2.get(i).y * (1 - HidList2.get(i).y) * weight_HO[i][j]
						* OutList.get(j).delta;
			}
		}
		for (int i = 0; i < HidList1.size(); i++) {
			HidList1.get(i).delta = 0;
			for (int j = 0; j < HidList2.size(); j++) {
				HidList1.get(i).delta += HidList1.get(i).y * (1 - HidList1.get(i).y) * weight_HH[i][j]
						* HidList2.get(j).delta;
			}
		}
	}

	void update_weight() {
		for (int i = 0; i < InList.size(); i++) {
			for (int j = 0; j < HidList1.size(); j++) {
				weight_IH[i][j] = WIH_HIST[i][j].get(WIH_HIST[i][j].size() - 1)
						+ LearningRate * HidList1.get(j).delta * InList.get(i).y;
				WIH_HIST[i][j].add(weight_IH[i][j]);
			}
		}

		for (int i = 0; i < HidList1.size(); i++) {
			for (int j = 0; j < HidList2.size(); j++) {
				weight_HH[i][j] = WHH_HIST[i][j].get(WHH_HIST[i][j].size() - 1)
						+ LearningRate * HidList2.get(j).delta * HidList1.get(i).y;
				WHH_HIST[i][j].add(weight_HH[i][j]);
			}
		}

		for (int i = 0; i < HidList2.size(); i++) {
			for (int j = 0; j < OutList.size(); j++) {
				weight_HO[i][j] = WHO_HIST[i][j].get(WHO_HIST[i][j].size() - 1)
						+ LearningRate * OutList.get(j).delta * HidList2.get(i).y;
				WHO_HIST[i][j].add(weight_HO[i][j]);
			}
		}

		for (int i = 0; i < InList.size(); i++) {
			for (int j = 0; j < HidList1.size(); j++) {
				bias_hidden1[i] = BH1_HIST[i].get(BH1_HIST[i].size() - 1) + LearningRate * HidList1.get(i).delta;
			}
			BH1_HIST[i].add(bias_hidden1[i]);
		}
		for (int i = 0; i < HidList1.size(); i++) {
			for (int j = 0; j < HidList2.size(); j++) {
				bias_hidden2[i] = BH2_HIST[i].get(BH2_HIST[i].size() - 1) + LearningRate * HidList2.get(i).delta;
			}
			BH2_HIST[i].add(bias_hidden2[i]);
		}
		for (int i = 0; i < OutList.size(); i++) {
			bias_output[i] = BO_HIST[i].get(BO_HIST[i].size() - 1) + LearningRate * OutList.get(i).delta;
			BO_HIST[i].add(bias_output[i]);
		}
	}

}