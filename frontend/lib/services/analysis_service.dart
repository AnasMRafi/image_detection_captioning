import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'api_service.dart';
import '../models/detection_result.dart';

/// Response model for combined analysis endpoint
class AnalysisResult {
  final String caption;
  final double captionConfidence;
  final List<DetectionResult> detections;
  final int detectionCount;
  final String combinedDescription;
  final double inferenceTimeMs;

  AnalysisResult({
    required this.caption,
    required this.captionConfidence,
    required this.detections,
    required this.detectionCount,
    required this.combinedDescription,
    required this.inferenceTimeMs,
  });

// async factory constructor
  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    return AnalysisResult(
      caption: json['caption'] ?? '',
      captionConfidence: (json['caption_confidence'] ?? 0).toDouble(),
      detections: (json['detections'] as List? ?? [])
          .map((d) => DetectionResult.fromJson(d))
          .toList(),
      detectionCount: json['detection_count'] ?? 0,
      combinedDescription: json['combined_description'] ?? '',
      inferenceTimeMs: (json['inference_time_ms'] ?? 0).toDouble(),
    );
  }
}

/// Service for combined analysis (detection + guided captioning)
class AnalysisService {
  bool _isInitialized = false;
  
  Future<bool> initialize() async {
    try {
      final response = await http.get(
        Uri.parse('${ApiConfig.baseUrl}/health'),
      ).timeout(const Duration(seconds: 5));
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        _isInitialized = data['status'] == 'healthy';
        return _isInitialized;
      }
      return false;
    } catch (e) {
      return false;
    }
  }
  
  Future<AnalysisResult?> analyzeImage(File imageFile, {double confidence = 0.5}) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('${ApiConfig.baseUrl}/analyze?confidence=$confidence'),
      );
      
      request.files.add(await http.MultipartFile.fromPath(
        'image',
        imageFile.path,
      ));
      
      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 30),
      );
      
      final response = await http.Response.fromStream(streamedResponse);
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true) {
          return AnalysisResult.fromJson(data);
        }
      }
      return null;
    } catch (e) {
      print('Analysis error: $e');
      return null;
    }
  }
}
