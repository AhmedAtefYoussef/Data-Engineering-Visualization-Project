/* The stage: one canvas, one particle per located building, drawn as a point of warm light.

   NYC.Stage.go(layoutName) moves the same particles to a new arrangement in two stages: first the buildings that
   leave fade, then those that stay travel. Add arrangements with NYC.Stage.layouts[name] = function (stage) {...},
   which returns { x: Float32Array, y: Float32Array, alpha: Float32Array, size?: Float32Array } in CSS pixels. */
(function () {
  "use strict";

  var NYC = window.NYC;
  var COLOR_NEW = [244, 185, 66], COLOR_PRESERVATION = [95, 212, 196];
  var FADE_MS = 280, MOVE_MS = 950, SPARK_PX = 14, INTRO_MS = 2200;

  function sprite(rgb) {
    var size = SPARK_PX * 2, canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    var context = canvas.getContext("2d");
    var glow = context.createRadialGradient(SPARK_PX, SPARK_PX, 0, SPARK_PX, SPARK_PX, SPARK_PX);
    var color = rgb.join(",");
    glow.addColorStop(0, "rgba(" + color + ",0.95)");
    glow.addColorStop(0.14, "rgba(" + color + ",0.6)");
    glow.addColorStop(0.4, "rgba(" + color + ",0.05)");
    glow.addColorStop(1, "rgba(" + color + ",0)");
    context.fillStyle = glow;
    context.fillRect(0, 0, size, size);
    return canvas;
  }

  var Stage = NYC.Stage = { layouts: {}, data: null, width: 0, height: 0, listeners: [] };

  Stage.layouts.city = function (stage) {
    var data = stage.data, n = data.n, x = new Float32Array(n), y = new Float32Array(n), alpha = new Float32Array(n);
    var lonMax = 0, latMax = 0, i;
    for (i = 0; i < n; i++) { lonMax = Math.max(lonMax, data.lon[i]); latMax = Math.max(latMax, data.lat[i]); }
    var midLat = (data.origin[1] + latMax / 2) / data.scale;
    var aspect = Math.cos(midLat * Math.PI / 180) * lonMax / latMax;
    var wide = stage.width > 960;
    var boxH = (wide ? 0.9 : 0.62) * stage.height, boxW = (wide ? 0.56 : 0.92) * stage.width;
    var drawH = Math.min(boxH, boxW / aspect), drawW = drawH * aspect;
    var left = wide ? stage.width * 0.97 - drawW : (stage.width - drawW) / 2;
    var top = wide ? (stage.height - drawH) / 2 : stage.height * 0.09;
    for (i = 0; i < n; i++) {
      x[i] = left + data.lon[i] / lonMax * drawW;
      y[i] = top + (1 - data.lat[i] / latMax) * drawH;
      alpha[i] = 1;
    }
    return { x: x, y: y, alpha: alpha };
  };

  NYC.onMount("#ov-stage", function (host) {
    var canvas = host.querySelector("canvas"), context = canvas.getContext("2d");
    var sprites = [sprite(COLOR_PRESERVATION), sprite(COLOR_NEW)];
    var scale = 1, current = null, from = null, to = null, startedAt = 0, seeds = null, visible = true, pointerX = 0, pointerY = 0, driftX = 0, driftY = 0;
    var layoutName = "city", intro = true, running = false;

    function resize() {
      scale = Math.min(window.devicePixelRatio || 1, 2);
      Stage.width = host.clientWidth; Stage.height = host.clientHeight;
      canvas.width = Math.round(Stage.width * scale); canvas.height = Math.round(Stage.height * scale);
      if (!Stage.data) { return; }
      var layout = Stage.layouts[layoutName](Stage);
      if (current) { current = copy(layout); from = to = null; } else { current = copy(layout); }
      kick();
    }
    function copy(layout) { return { x: layout.x.slice(), y: layout.y.slice(), alpha: layout.alpha.slice(), size: layout.size ? layout.size.slice() : null }; }

    function draw(now) {
      running = false;
      if (!Stage.data || !current) { return; }
      var n = Stage.data.n, i, more = false, reduced = NYC.reducedMotion();
      if (to) {
        var elapsed = now - startedAt, fade = Math.min(1, elapsed / FADE_MS), move = NYC.easeOut(Math.min(1, Math.max(0, (elapsed - FADE_MS) / MOVE_MS)));
        for (i = 0; i < n; i++) {
          var leaving = to.alpha[i] < from.alpha[i];
          current.alpha[i] = leaving ? from.alpha[i] + (to.alpha[i] - from.alpha[i]) * fade : from.alpha[i] + (to.alpha[i] - from.alpha[i]) * move;
          current.x[i] = from.x[i] + (to.x[i] - from.x[i]) * move;
          current.y[i] = from.y[i] + (to.y[i] - from.y[i]) * move;
        }
        if (elapsed >= FADE_MS + MOVE_MS) { from = to = null; } else { more = true; }
      }
      driftX += (pointerX - driftX) * 0.06; driftY += (pointerY - driftY) * 0.06;
      if (Math.abs(pointerX - driftX) > 0.2 || Math.abs(pointerY - driftY) > 0.2) { more = true; }

      context.setTransform(scale, 0, 0, scale, 0, 0);
      context.clearRect(0, 0, Stage.width, Stage.height);
      context.globalCompositeOperation = "lighter";
      var introT = intro && !reduced ? Math.min(1, (now - seeds.start) / INTRO_MS) : 1;
      for (i = 0; i < n; i++) {
        var a = current.alpha[i];
        if (a <= 0.01) { continue; }
        var shown = introT < 1 ? Math.max(0, Math.min(1, (introT - seeds.delay[i]) / 0.25)) : 1;
        if (shown <= 0) { continue; }
        // Lights come on across the city, each with a brief flare. Buildings with more homes burn a little larger.
        var flare = introT < 1 ? 1 + (1 - shown) * 1.6 : 1;
        var size = (current.size ? current.size[i] : seeds.size[i]) * flare;
        var twinkle = reduced ? 1 : 0.86 + 0.14 * Math.sin(now * seeds.rate[i] + seeds.phase[i]);
        context.globalAlpha = a * shown * twinkle;
        context.drawImage(sprites[Stage.data.new[i]], current.x[i] + driftX - size / 2, current.y[i] + driftY - size / 2, size, size);
      }
      context.globalAlpha = 1;
      if (introT >= 1) { intro = false; }
      if (more || intro || (visible && !reduced)) { kick(); }
    }
    function kick() { if (running || !visible) { return; } running = true; window.requestAnimationFrame(draw); }

    Stage.go = function (name) {
      if (!Stage.layouts[name] || !current) { return; }
      layoutName = name;
      var target = copy(Stage.layouts[name](Stage));
      if (NYC.reducedMotion()) { current = target; from = to = null; kick(); return; }
      from = copy(current); to = target; startedAt = performance.now();
      if (target.size) { current.size = target.size; } else { current.size = null; }
      kick();
    };

    new ResizeObserver(resize).observe(host);
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) { visible = entries[0].isIntersecting; if (visible) { kick(); } }).observe(host);
    }
    if (window.matchMedia("(pointer: fine)").matches) {
      window.addEventListener("pointermove", function (event) {
        pointerX = (event.clientX / window.innerWidth - 0.5) * -14; pointerY = (event.clientY / window.innerHeight - 0.5) * -10;
        kick();
      }, { passive: true });
    }

    fetch(host.getAttribute("data-url")).then(function (response) {
      if (!response.ok) { throw new Error("story data " + response.status); }
      return response.json();
    }).then(function (data) {
      Stage.data = data;
      var n = data.n, i;
      seeds = { size: new Float32Array(n), rate: new Float32Array(n), phase: new Float32Array(n), delay: new Float32Array(n), start: performance.now() };
      for (i = 0; i < n; i++) {
        seeds.size[i] = Math.min(10, 4.5 + Math.log10(1 + data.units[i]) * 1.8);
        seeds.rate[i] = 0.0006 + Math.random() * 0.0016;
        seeds.phase[i] = Math.random() * 6.28;
        seeds.delay[i] = Math.random() * 0.75;
      }
      resize();
      Stage.listeners.forEach(function (listener) { listener(data); });
    }).catch(function (error) {
      host.setAttribute("data-failed", String(error));    // the text beside it stands without the canvas
    });
  });
})();
