/* The Generate report sequence. When the reader asks for a report, the whole screen goes dark, the points of the
   city stream in as real steps of the work are named, and when the report is ready a circle opens onto it. Dash's
   own `running` state on #report (class is-loading) says when the work starts and ends; a short minimum keeps the
   sequence readable even when the answer is instant. Not shown for the report drawn on page load. */
(function () {
  "use strict";

  var NYC = window.NYC;
  var MIN_MS = 2300, STEP_MS = 620, MAX_WAIT_MS = 15000, OUT_MS = 900, POINTS = 1800;
  var armed = false, overlay = null, raf = 0, startedAt = 0, finishing = false, done = false, fallback = 0;

  function steps() {
    var facts = document.getElementById("facts"), d = facts ? facts.dataset : {};
    return ["Reading " + (d.crashes || "") + " crash reports",
            "Placing " + (d.sites || "") + " buildings on the map",
            "Measuring the streets within 250 m of each one",
            "Comparing before and after"];
  }

  function points(width, height) {
    var out = [], data = NYC.Stage && NYC.Stage.data, count = data ? Math.min(POINTS, data.n) : POINTS, i;
    var lonMax = 1, latMax = 1;
    if (data) { for (i = 0; i < data.n; i++) { lonMax = Math.max(lonMax, data.lon[i]); latMax = Math.max(latMax, data.lat[i]); } }
    var drawH = height * 0.88, drawW = drawH * 0.82, stride = data ? data.n / count : 1;
    for (i = 0; i < count; i++) {
      var angle = Math.random() * 6.283, radius = Math.max(width, height) * (0.6 + Math.random() * 0.5), index = Math.floor(i * stride);
      out.push({
        sx: width / 2 + Math.cos(angle) * radius, sy: height / 2 + Math.sin(angle) * radius,
        tx: data ? width / 2 - drawW / 2 + data.lon[index] / lonMax * drawW : width / 2 + Math.cos(angle) * height * 0.32,
        ty: data ? height / 2 + drawH / 2 - data.lat[index] / latMax * drawH : height / 2 + Math.sin(angle) * height * 0.32,
        hot: data ? data.new[index] : 1, delay: Math.random() * 0.45, spin: 0.6 + Math.random() * 1.2
      });
    }
    return out;
  }

  function open() {
    if (overlay) { return; }
    done = finishing = false;
    startedAt = performance.now();
    overlay = document.createElement("div");
    overlay.className = "gen";
    overlay.setAttribute("role", "status");
    overlay.setAttribute("aria-live", "polite");
    var canvas = document.createElement("canvas");
    canvas.className = "gen-canvas"; canvas.setAttribute("aria-hidden", "true");
    var panel = document.createElement("div");
    panel.className = "gen-panel";
    var title = document.createElement("p"), line = document.createElement("p"), bar = document.createElement("div");
    title.className = "gen-title"; title.textContent = "Building your report";
    line.className = "gen-line"; line.textContent = steps()[0];
    bar.className = "gen-bar"; bar.appendChild(document.createElement("i"));
    panel.appendChild(title); panel.appendChild(line); panel.appendChild(bar);
    overlay.appendChild(canvas); overlay.appendChild(panel);
    document.body.appendChild(overlay);
    document.documentElement.style.overflow = "hidden";

    var reduced = NYC.reducedMotion(), texts = steps(), stepShown = 0;
    var context = canvas.getContext("2d"), width = window.innerWidth, height = window.innerHeight, scale = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * scale; canvas.height = height * scale;
    context.scale(scale, scale);
    var cloud = reduced ? [] : points(width, height);

    function frame(now) {
      var elapsed = now - startedAt, form = NYC.easeOut(Math.min(1, elapsed / (MIN_MS * 0.8)));
      var step = Math.min(texts.length - 1, Math.floor(elapsed / STEP_MS));
      if (step !== stepShown) {
        stepShown = step;
        line.textContent = texts[step];
        line.classList.remove("is-new"); void line.offsetWidth; line.classList.add("is-new");
      }
      bar.firstChild.style.setProperty("--p", Math.min(done ? 1 : 0.94, elapsed / MIN_MS).toFixed(3));
      if (!reduced) {
        context.clearRect(0, 0, width, height);
        context.globalCompositeOperation = "lighter";
        cloud.forEach(function (p) {
          var t = NYC.easeOut(Math.max(0, Math.min(1, (form - p.delay) / (1 - p.delay))));
          var swirl = (1 - t) * p.spin * 2.2;
          var dx = p.sx - p.tx, dy = p.sy - p.ty, cos = Math.cos(swirl), sin = Math.sin(swirl);
          var x = p.tx + (dx * cos - dy * sin) * (1 - t), y = p.ty + (dx * sin + dy * cos) * (1 - t);
          context.fillStyle = p.hot ? "rgba(244,185,66," : "rgba(95,212,196,";
          context.fillStyle += (0.25 + 0.75 * t).toFixed(2) + ")";
          context.fillRect(x, y, 1.6, 1.6);
        });
        // A ring of light sweeps out from the middle, the 250 m circle at city scale.
        var ring = (elapsed % 1600) / 1600;
        context.strokeStyle = "rgba(255,111,89," + (0.4 * (1 - ring)).toFixed(3) + ")";
        context.lineWidth = 1.5;
        context.beginPath(); context.arc(width / 2, height / 2, ring * Math.max(width, height) * 0.6, 0, 6.283); context.stroke();
      }
      maybeFinish(now);
      if (overlay) { raf = window.requestAnimationFrame(frame); }
    }
    raf = window.requestAnimationFrame(frame);
    fallback = window.setTimeout(function () { done = true; }, MAX_WAIT_MS);   // the page never stays covered
  }

  function maybeFinish(now) {
    if (!done || finishing || now - startedAt < MIN_MS) { return; }
    finishing = true;
    var target = document.getElementById("report");
    if (target) { target.scrollIntoView({ block: "start", behavior: "instant" }); }
    document.documentElement.style.overflow = "";
    document.querySelectorAll(".tile, .card").forEach(function (element, index) { element.style.setProperty("--i", index % 8); });
    document.body.setAttribute("data-report", "fresh");
    window.setTimeout(function () { document.body.removeAttribute("data-report"); }, 2400);
    var leaving = overlay;
    leaving.classList.add("is-out");
    window.setTimeout(function () {
      window.cancelAnimationFrame(raf);
      leaving.remove();
      overlay = null;
    }, NYC.reducedMotion() ? 0 : OUT_MS);
  }

  /* The reader asked: from the button, or Enter in the search box. */
  document.addEventListener("click", function (event) {
    if (event.target.closest && event.target.closest("#generate")) { armed = true; open(); }
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && event.target.id === "search") { armed = true; open(); }
  });

  NYC.onMount("#report", function (report) {
    var wasLoading = report.classList.contains("is-loading");
    new MutationObserver(function () {
      var loading = report.classList.contains("is-loading");
      if (wasLoading && !loading && armed) {
        armed = false;
        window.clearTimeout(fallback);
        done = true;
      }
      wasLoading = loading;
    }).observe(report, { attributes: true, attributeFilter: ["class"] });
  });
})();
