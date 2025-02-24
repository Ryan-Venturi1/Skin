/**
 * Angle Tracking Module
 * Provides dynamic suggestions based on image analysis for better camera alignment.
 * For now, returns random suggestions; this can be enhanced with more advanced analysis.
 */
function getAngleSuggestion(imageElement) {
    const suggestions = [
      "Move camera left for a better view.",
      "Move camera right for a better view.",
      "Move camera up for a better view.",
      "Move camera down for a better view.",
      "Hold steady for a clear image."
    ];
    return suggestions[Math.floor(Math.random() * suggestions.length)];
  }
  
  // Expose the function globally
  window.getAngleSuggestion = getAngleSuggestion;