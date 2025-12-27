/// Detection result model for object detection API response
class DetectionResult {
  /// Bounding box coordinates [x1, y1, x2, y2]
  final List<double> box;
  
  /// Class label index
  final int label;
  
  /// Confidence score (0.0 - 1.0)
  final double score;
  
  /// Human-readable class name (e.g., "person", "car")
  final String className;
  
  DetectionResult({
    required this.box,
    required this.label,
    required this.score,
    required this.className,
  });
  
  /// Create from JSON response
  factory DetectionResult.fromJson(Map<String, dynamic> json) {
    return DetectionResult(
      box: (json['box'] as List).map((e) => (e as num).toDouble()).toList(),
      label: json['label'] as int,
      score: (json['score'] as num).toDouble(),
      className: json['class_name'] as String,
    );
  }
  
  /// Get bounding box as [left, top, right, bottom]
  double get left => box[0];
  double get top => box[1];
  double get right => box[2];
  double get bottom => box[3];
  
  /// Get bounding box width and height
  double get width => right - left;
  double get height => bottom - top;
  
  /// Format score as percentage
  String get scorePercent => '${(score * 100).toStringAsFixed(0)}%';
  
  @override
  String toString() => 'DetectionResult($className: $scorePercent)';
}
