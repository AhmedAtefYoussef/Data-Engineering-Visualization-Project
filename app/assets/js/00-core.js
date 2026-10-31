/* Shared browser helpers, as window.NYC. Views are mounted by React after the first paint, so nothing here
   assumes an element exists at load:

     NYC.onMount(selector, fn)   runs fn(element) once for each matching element that appears; fn may return a cleanup
     NYC.reducedMotion()         true when the reader asked for less motion
     NYC.countUp(element)        counts a [data-countup] figure up to its value, ending on the exact text
     NYC.onRoute(path, titles)   called by the route callback: marks the rail link, names the tab, goes to the top */
(function () {
  "use strict";

  var NYC = window.NYC = window.NYC || {};
  var COUNT_MS = 700;
  var motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  NYC.reducedMotion = function () { return motion.matches; };
  NYC.easeOut = function (t) { return 1 - Math.pow(1 - t, 4); };   // the same settle as --ease-out

  var watchers = [];
  var mounted = new WeakMap();

  function scan() {
    watchers.forEach(function (watcher) {
      document.querySelectorAll(watcher.selector).forEach(function (element) {
        var seen = mounted.get(element) || [];
        if (seen.indexOf(watcher) >= 0) { return; }
        seen.push(watcher);
        mounted.set(element, seen);
        watcher.fn(element);
      });
    });
  }

  NYC.onMount = function (selector, fn) {
    watchers.push({ selector: selector, fn: fn });
    scan();
  };

  var pending = false;
  function schedule() {
    if (pending) { return; }
    pending = true;
    window.setTimeout(function () { pending = false; scan(); }, 40);   // a timer, not rAF: rAF stops in a hidden tab
  }

  // Keeps the shape of the final text: a sign, separators and a suffix stay as they are.
  function formatLike(final, value) {
    var match = final.match(/^([^\d]*)([\d,]*)(\.\d+)?(.*)$/);
    if (!match) { return final; }
    var decimals = match[3] ? match[3].length - 1 : 0;
    var body = Math.abs(value).toFixed(decimals);
    var parts = body.split(".");
    if (match[2].indexOf(",") >= 0) { parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
    return match[1] + parts.join(".") + match[4];
  }

  NYC.countUp = function (element) {
    var final = element.getAttribute("data-final") || element.textContent;
    var target = parseFloat(element.getAttribute("data-countup"));
    if (NYC.reducedMotion() || !isFinite(target)) { element.textContent = final; return; }
    var start = null;
    function frame(now) {
      if (start === null) { start = now; }
      var t = Math.min(1, (now - start) / COUNT_MS);
      element.textContent = t < 1 ? formatLike(final, target * NYC.easeOut(t)) : final;
      if (t < 1) { window.requestAnimationFrame(frame); }
    }
    window.requestAnimationFrame(frame);
  };

  NYC.onRoute = function (pathname, titles) {
    var path = (pathname || "/").replace(/\/+$/, "") || "/";
    document.querySelectorAll(".nav-link").forEach(function (link) {
      if (link.getAttribute("href") === path) { link.setAttribute("aria-current", "page"); }
      else { link.removeAttribute("aria-current"); }
    });
    if (titles && titles[path]) { document.title = titles[path] + " · NYC housing and street safety"; }
    window.scrollTo(0, 0);
    return path;
  };

  function start() {
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
    scan();
  }

  NYC.onMount("[data-countup]", NYC.countUp);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
