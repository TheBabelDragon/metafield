/**
 * Observation history, playback, scrubbing.
 */
export function createTimeline({ onFrame, onIndexChange }) {
  let history = [];
  let index = 0;
  let playing = false;
  let demoMode = false;
  let accum = 0;
  const frameDuration = 0.55; // seconds per tick in autoplay / demo

  function load(sequence) {
    history = sequence.slice();
    index = 0;
    playing = false;
    demoMode = false;
    emit();
  }

  function emit() {
    const obs = history[index] ?? null;
    if (onFrame) onFrame(obs, index, history.length);
    if (onIndexChange) onIndexChange(index, history.length);
  }

  function setIndex(i) {
    if (history.length === 0) return;
    index = Math.max(0, Math.min(history.length - 1, i | 0));
    emit();
  }

  function play() {
    playing = true;
    demoMode = false;
  }

  function pause() {
    playing = false;
    demoMode = false;
  }

  function toggleDemo(on) {
    demoMode = on !== undefined ? on : !demoMode;
    playing = demoMode;
    if (demoMode && history.length === 0) return;
  }

  function isDemo() {
    return demoMode;
  }

  function isPlaying() {
    return playing;
  }

  function reset() {
    index = 0;
    playing = false;
    demoMode = false;
    accum = 0;
    emit();
  }

  function update(dt) {
    if (!playing || history.length === 0) return;
    accum += dt;
    if (accum >= frameDuration) {
      accum = 0;
      if (index < history.length - 1) {
        index += 1;
        emit();
      } else if (demoMode) {
        // Loop demo
        index = 0;
        emit();
      } else {
        playing = false;
      }
    }
  }

  function getHistory() {
    return history;
  }

  function getIndex() {
    return index;
  }

  return {
    load,
    setIndex,
    play,
    pause,
    toggleDemo,
    isDemo,
    isPlaying,
    reset,
    update,
    getHistory,
    getIndex,
  };
}
