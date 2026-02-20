import 'package:flutter/material.dart';

class GanInfoPage extends StatelessWidget {
  const GanInfoPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("GAN Architecture"),
        backgroundColor: Colors.deepPurple,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView(
          children: const [
            Text(
              "Vanilla GAN for Signature Generation",
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 20),

            Text(
              "Generator (G)",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            Text(
              "Input: Latent Vector z (100-dim)\n"
              "Dense Layer → Reshape (4x4x256)\n"
              "ConvTranspose + BatchNorm + ReLU\n"
              "Upsampling Blocks\n"
              "Final Layer: Conv2D (1 channel) + Tanh\n"
              "Output: 64x64 Grayscale Signature Image",
              style: TextStyle(fontSize: 15),
            ),

            SizedBox(height: 20),

            Text(
              "Discriminator (D)",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            Text(
              "Input: 64x64 Signature Image\n"
              "Conv2D + LeakyReLU(0.2)\n"
              "Downsampling Layers\n"
              "Flatten + Dense(1)\n"
              "Output: Real / Fake Probability",
              style: TextStyle(fontSize: 15),
            ),

            SizedBox(height: 20),

            Text(
              "Training Details",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            Text(
              "Loss Function: Binary Cross Entropy (BCE)\n"
              "Optimizer: Adam (lr = 0.0002, β1 = 0.5)\n"
              "Epochs: 200\n"
              "Dataset: Handwritten Signatures (5 Users × 20 Samples)",
              style: TextStyle(fontSize: 15),
            ),

            SizedBox(height: 20),

            Text(
              "Project Purpose",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            Text(
              "This GAN generates synthetic handwritten signatures "
              "to augment datasets for signature verification systems "
              "used in banking, exams, and identity verification.",
              style: TextStyle(fontSize: 15),
            ),
          ],
        ),
      ),
    );
  }
}