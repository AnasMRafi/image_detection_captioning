/// Caption result model for API response
class CaptionResult {
  final String caption;
  final double confidence;
  final double inferenceTimeMs;
  final List<AlternativeCaption>? alternatives;
  final DateTime timestamp;

  CaptionResult({
    required this.caption,
    required this.confidence,
    required this.inferenceTimeMs,
    this.alternatives,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  factory CaptionResult.fromJson(Map<String, dynamic> json) {
    return CaptionResult(
      caption: json['caption'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      inferenceTimeMs: (json['inference_time_ms'] as num).toDouble(),
      alternatives: json['alternatives'] != null
          ? (json['alternatives'] as List)
              .map((e) => AlternativeCaption.fromJson(e))
              .toList()
          : null,
    );
  }

  Map<String, dynamic> toJson() => {
        'caption': caption,
        'confidence': confidence,
        'inference_time_ms': inferenceTimeMs,
        'alternatives': alternatives?.map((e) => e.toJson()).toList(),
        'timestamp': timestamp.toIso8601String(),
      };
}

/// Alternative caption from beam search
class AlternativeCaption {
  final String caption;
  final double score;

  AlternativeCaption({
    required this.caption,
    required this.score,
  });

  factory AlternativeCaption.fromJson(Map<String, dynamic> json) {
    return AlternativeCaption(
      caption: json['caption'] as String,
      score: (json['score'] as num).toDouble(),
    );
  }

  Map<String, dynamic> toJson() => {
        'caption': caption,
        'score': score,
      };
}
