import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/caption_result.dart';

import 'package:shared_preferences/shared_preferences.dart';

/// API configuration for the captioning backend
class ApiConfig {
  static const String _prefKey = 'api_base_url';
  
  /// Default URL
  static const String _defaultUrl = 'http://10.40.17.94:8000';
  
  /// Current Base URL
  static String _baseUrl = _defaultUrl;
  
  static String get baseUrl => _baseUrl;
  
  /// Initialize configuration from SharedPreferences
  static Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    // TEMPORARY: Force reset to new IP (remove this after first run)
    await prefs.remove(_prefKey);
    _baseUrl = prefs.getString(_prefKey) ?? _defaultUrl;
  }
  
  /// Update and save the Base URL
  static Future<void> setBaseUrl(String newUrl) async {
    // Ensure protocol is present
    if (!newUrl.startsWith('http')) {
      newUrl = 'http://$newUrl';
    }
    // Remove trailing slash
    if (newUrl.endsWith('/')) {
      newUrl = newUrl.substring(0, newUrl.length - 1);
    }
    
    _baseUrl = newUrl;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefKey, _baseUrl);
  }
  
  /// Timeout for API requests (seconds)
  /// Note: First inference can take 60-90s due to MPS/GPU compilation
  static const int timeoutSeconds = 120;
}

/// Service for communicating with the image captioning backend
class ApiService {
  final http.Client _client;
  
  ApiService({http.Client? client}) : _client = client ?? http.Client();
  
  /// Check if the backend server is healthy and ready
  Future<bool> healthCheck() async {
    try {
      final response = await _client
          .get(Uri.parse('${ApiConfig.baseUrl}/health'))
          .timeout(const Duration(seconds: 5));
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['status'] == 'healthy' && data['model_loaded'] == true;
      }
      return false;
    } catch (e) {
      print('Health check failed: $e');
      return false;
    }
  }
  
  /// Generate a caption for an image file
  /// 
  /// [imageFile] - The image file to caption
  /// 
  /// Returns a [CaptionResult] with the generated caption and confidence
  /// 
  /// Throws [ApiException] if the request fails
  Future<CaptionResult> captionImage(File imageFile) async {
    try {
      // Create multipart request
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('${ApiConfig.baseUrl}/caption'),
      );
      
      // Add image file to request
      request.files.add(
        await http.MultipartFile.fromPath(
          'image',
          imageFile.path,
        ),
      );
      
      // Send request with timeout
      final streamedResponse = await request.send().timeout(
        const Duration(seconds: ApiConfig.timeoutSeconds),
      );
      
      // Read response body
      final response = await http.Response.fromStream(streamedResponse);
      
      // Parse response
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        
        if (data['success'] == true) {
          return CaptionResult.fromJson(data);
        } else {
          throw ApiException(
            message: data['error'] ?? 'Unknown error',
            code: data['code'] ?? 'UNKNOWN_ERROR',
          );
        }
      } else {
        final data = jsonDecode(response.body);
        throw ApiException(
          message: data['error'] ?? 'Server error',
          code: data['code'] ?? 'SERVER_ERROR',
          statusCode: response.statusCode,
        );
      }
    } on ApiException {
      rethrow;
    } catch (e) {
      if (e.toString().contains('timeout')) {
        throw ApiException(
          message: 'Request timed out. Please try again.',
          code: 'TIMEOUT',
        );
      } else if (e.toString().contains('SocketException') || 
                 e.toString().contains('Connection refused')) {
        throw ApiException(
          message: 'Cannot connect to server. Make sure the backend is running.',
          code: 'CONNECTION_ERROR',
        );
      }
      throw ApiException(
        message: 'Failed to process image: $e',
        code: 'UNKNOWN_ERROR',
      );
    }
  }
  
  /// Generate caption from image bytes (for camera captures)
  Future<CaptionResult> captionImageBytes(List<int> bytes, String filename) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('${ApiConfig.baseUrl}/caption'),
      );
      
      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          bytes,
          filename: filename,
        ),
      );
      
      final streamedResponse = await request.send().timeout(
        const Duration(seconds: ApiConfig.timeoutSeconds),
      );
      
      final response = await http.Response.fromStream(streamedResponse);
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true) {
          return CaptionResult.fromJson(data);
        } else {
          throw ApiException(
            message: data['error'] ?? 'Unknown error',
            code: data['code'] ?? 'UNKNOWN_ERROR',
          );
        }
      } else {
        throw ApiException(
          message: 'Server error: ${response.statusCode}',
          code: 'SERVER_ERROR',
          statusCode: response.statusCode,
        );
      }
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException(
        message: 'Failed to process image: $e',
        code: 'UNKNOWN_ERROR',
      );
    }
  }
  
  /// Dispose the HTTP client
  void dispose() {
    _client.close();
  }
}

/// Custom exception for API errors
class ApiException implements Exception {
  final String message;
  final String code;
  final int? statusCode;
  
  ApiException({
    required this.message,
    required this.code,
    this.statusCode,
  });
  
  @override
  String toString() => 'ApiException: [$code] $message';
}
