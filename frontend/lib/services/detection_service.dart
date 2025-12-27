import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'api_service.dart';
import '../models/detection_result.dart';

/// Service for server-side object detection using Faster R-CNN
class DetectionService {
  final http.Client _client;
  bool _isInitialized = false;
  
  DetectionService({http.Client? client}) : _client = client ?? http.Client();
  
  /// Check if detection service is available
  bool get isInitialized => _isInitialized;
  
  /// Initialize the detection service (check server health)
  Future<bool> initialize() async {
    try {
      final response = await _client
          .get(Uri.parse('${ApiConfig.baseUrl}/health'))
          .timeout(const Duration(seconds: 5));
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        _isInitialized = data['models_loaded']?['detector'] == true;
        return _isInitialized;
      }
      return false;
    } catch (e) {
      print('Detection service init failed: $e');
      _isInitialized = false;
      return false;
    }
  }
  
  /// Detect objects in an image file
  /// 
  /// [imageFile] - The image file to analyze
  /// [confidenceThreshold] - Minimum confidence (0.0 - 1.0)
  /// 
  /// Returns a list of [DetectionResult] with bounding boxes
  Future<List<DetectionResult>> detectFromFile(
    File imageFile, {
    double confidenceThreshold = 0.5,
  }) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('${ApiConfig.baseUrl}/detect?confidence=$confidenceThreshold'),
      );
      
      request.files.add(
        await http.MultipartFile.fromPath('image', imageFile.path),
      );
      
      final streamedResponse = await request.send().timeout(
        const Duration(seconds: ApiConfig.timeoutSeconds),
      );
      
      final response = await http.Response.fromStream(streamedResponse);
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true) {
          final detections = (data['detections'] as List)
              .map((d) => DetectionResult.fromJson(d))
              .toList();
          return detections;
        }
      }
      
      return [];
    } catch (e) {
      print('Detection error: $e');
      return [];
    }
  }
  
  /// Detect objects from image bytes (for camera frames)
  /// 
  /// [bytes] - Image bytes
  /// [filename] - Filename with extension (e.g., 'frame.jpg')
  /// [confidenceThreshold] - Minimum confidence (0.0 - 1.0)
  Future<List<DetectionResult>> detectFromBytes(
    Uint8List bytes, {
    String filename = 'frame.jpg',
    double confidenceThreshold = 0.5,
  }) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('${ApiConfig.baseUrl}/detect?confidence=$confidenceThreshold'),
      );
      
      request.files.add(
        http.MultipartFile.fromBytes('image', bytes, filename: filename),
      );
      
      final streamedResponse = await request.send().timeout(
        const Duration(seconds: ApiConfig.timeoutSeconds),
      );
      
      final response = await http.Response.fromStream(streamedResponse);
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true) {
          final detections = (data['detections'] as List)
              .map((d) => DetectionResult.fromJson(d))
              .toList();
          return detections;
        }
      }
      
      return [];
    } catch (e) {
      print('Detection error: $e');
      return [];
    }
  }
  
  /// Dispose resources
  void dispose() {
    _client.close();
    _isInitialized = false;
  }
}
