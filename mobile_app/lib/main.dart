import 'package:flutter/material.dart';
import 'dart:typed_data'; // 🔥 REQUIRED for Uint8List
import 'services/api_service.dart';
import 'gan_info_page.dart'; // GAN architecture screen

void main() {
  runApp(const SignatureGANApp());
}

class SignatureGANApp extends StatelessWidget {
  const SignatureGANApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Synthetic Signature Generator',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        scaffoldBackgroundColor: const Color(0xFFF5F3F8),
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => HomePageState();
}

class HomePageState extends State<HomePage> {
  int numSignatures = 16;
  bool isLoading = false;
  Uint8List? imageBytes; // 🔥 Stores image from backend

  Future<void> generateSignatures() async {
    setState(() {
      isLoading = true;
    });

    try {
      final bytes =
          await ApiService.generateSignatures(numSignatures);

      setState(() {
        imageBytes = bytes;
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        isLoading = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error: $e")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Synthetic Signature Generator"),
        centerTitle: true,
        elevation: 4,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            /// 🔷 CONTROL CARD
            Card(
              elevation: 5,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    const Text(
                      "Generation Controls",
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 12),

                    /// Slider for number of signatures
                    Row(
                      mainAxisAlignment:
                          MainAxisAlignment.spaceBetween,
                      children: [
                        const Text("Number of Signatures:"),
                        Text(
                          "$numSignatures",
                          style: const TextStyle(
                              fontWeight: FontWeight.bold),
                        ),
                      ],
                    ),
                    Slider(
                      min: 4,
                      max: 32,
                      divisions: 7,
                      value: numSignatures.toDouble(),
                      onChanged: (value) {
                        setState(() {
                          numSignatures = value.toInt();
                        });
                      },
                    ),

                    const SizedBox(height: 10),

                    /// Generate Button
                    ElevatedButton.icon(
                      onPressed: isLoading
                          ? null
                          : generateSignatures,
                      icon: const Icon(Icons.auto_awesome),
                      label:
                          const Text("Generate Signatures"),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 24, vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius:
                              BorderRadius.circular(12),
                        ),
                      ),
                    ),

                    const SizedBox(height: 10),

                    /// GAN Architecture Button
                    OutlinedButton.icon(
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) =>
                                const GanInfoPage(),
                          ),
                        );
                      },
                      icon: const Icon(Icons.info_outline),
                      label:
                          const Text("View GAN Architecture"),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 20),

            /// 🔷 OUTPUT SECTION
            Expanded(
              child: Center(
                child: isLoading
                    ? Column(
                        mainAxisAlignment:
                            MainAxisAlignment.center,
                        children: const [
                          CircularProgressIndicator(),
                          SizedBox(height: 12),
                          Text(
                            "Generating synthetic signatures...",
                            style: TextStyle(fontSize: 16),
                          ),
                        ],
                      )
                    : imageBytes != null
                        ? Column(
                            children: [
                              const Text(
                                "Generated Signatures",
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight:
                                      FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 12),
                              Expanded(
                                child: SingleChildScrollView(
                                  child: Image.memory(
                                    imageBytes!,
                                    fit: BoxFit.contain,
                                  ),
                                ),
                              ),
                            ],
                          )
                        : const Text(
                            "Generated signatures will appear here",
                            style: TextStyle(fontSize: 16),
                          ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}