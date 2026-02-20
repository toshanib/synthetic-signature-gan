import 'dart:typed_data';
import 'package:http/http.dart' as http;

class ApiService {
  // Android Emulator localhost bridge
  static const String baseUrl = "http://10.0.2.2:8000";

  static Future<Uint8List> generateSignatures(int n) async {
    final uri = Uri.parse("$baseUrl/generate?n=$n");

    final response = await http.get(uri); // 🔥 FIX: GET not POST

    if (response.statusCode == 200) {
      return response.bodyBytes; // Direct image bytes (FAST)
    } else {
      throw Exception("Failed to generate signatures: ${response.statusCode}");
    }
  }
}