/* The page's motion, as window.NYC. Text stays plain and readable until this script improves it, and it does nothing
   but count-free final states under prefers-reduced-motion.

     NYC.onMount(selector, fn)  runs fn(element) once per matching element, whenever it appears (Dash mounts late)
     NYC.reducedMotion()        true when the reader asked for less motion
     NYC.easeOut(t)             the page's settle curve
     NYC.countUp(element)       counts a [data-countup] figure to its value, ending exactly on its text

   Declared in the markup:
     data-kinetic="words"       words rise out of a mask when the element enters the window
     data-kinetic="lines"       the element lifts in
     data-scrub="words"         words light up with the scroll position
     data-wave                  a wave runs through the letters on hover
     .magnetic                  leans toward the pointer; a press sends a ripple
     .spot                      a soft light follows the pointer
     data-tint on a section     the page's ground drifts to that tint while it is in view */
(function () {
  "use strict";

  var NYC = window.NYC = window.NYC || {};
  var motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  var fine = window.matchMedia("(hover: hover) and (pointer: fine)");
  var COUNT_MS = 1400, MAGNET = 0.28, SCRUB_LOW = 0.2, SCRUB_SOFT = 5;
  var TINTS = { midnight: "#0a0d1a", harbor: "#071620", plum: "#150d24", ember: "#1a0f14" };

  NYC.reducedMotion = function () { return motion.matches; };
  NYC.easeOut = function (t) { return 1 - Math.pow(1 - t, 4); };

  var watchers = [], mounted = new WeakMap(), pending = false;
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
  NYC.onMount = function (selector, fn) { watchers.push({ selector: selector, fn: fn }); scan(); };
  function schedule() {
    if (pending) { return; }
    pending = true;
    window.setTimeout(function () { pending = false; scan(); }, 40);   // a timer, not rAF: rAF stops in a hidden tab
  }

  /* Enter the window once. */
  var entrance = "IntersectionObserver" in window ? new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) { return; }
      entrance.unobserve(entry.target);
      entry.target.classList.add("is-in");
      if (entry.target.__onEnter) { entry.target.__onEnter(); }
    });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 }) : null;
  function whenSeen(element, fn) {
    if (!entrance) { element.classList.add("is-in"); if (fn) { fn(); } return; }
    if (fn) { element.__onEnter = fn; }
    entrance.observe(element);
  }

  /* Replace an element's text with spans. Its accessible name stays the whole sentence. */
  function splitWords(element, wrap) {
    var text = element.textContent;
    if (!text.trim() || element.children.length) { return null; }
    element.setAttribute("aria-label", text);
    if (!element.getAttribute("role") && element.tagName === "P") { element.setAttribute("role", "paragraph"); }
    element.textContent = "";
    var words = [];
    text.trim().split(/\s+/).forEach(function (word, index, all) {
      var node = wrap(word, index);
      node.setAttribute("aria-hidden", "true");
      words.push(node);
      element.appendChild(node);
      if (index < all.length - 1) { element.appendChild(document.createTextNode(" ")); }
    });
    return words;
  }

  NYC.onMount("[data-kinetic='words']", function (element) {
    if (NYC.reducedMotion()) { return; }
    var done = splitWords(element, function (word, index) {
      var outer = document.createElement("span"), inner = document.createElement("span");
      outer.className = "kw"; inner.className = "kw-i";
      inner.style.setProperty("--i", index);
      inner.textContent = word;
      outer.appendChild(inner);
      return outer;
    });
    if (done) { whenSeen(element); }
  });

  NYC.onMount("[data-kinetic='lines']", function (element) {
    if (NYC.reducedMotion()) { return; }
    element.classList.add("lift");
    whenSeen(element);
  });
  NYC.onMount(".lift", function (element) { whenSeen(element); });

  /* Words that light up as the paragraph moves through the window. */
  var scrubbing = [];
  NYC.onMount("[data-scrub='words']", function (element) {
    if (NYC.reducedMotion()) { return; }
    var words = splitWords(element, function (word) {
      var span = document.createElement("span");
      span.className = "sw"; span.textContent = word;
      return span;
    });
    if (words) { scrubbing.push({ element: element, words: words }); onScroll(); }
  });
  function updateScrub(height) {
    scrubbing.forEach(function (item) {
      var box = item.element.getBoundingClientRect();
      if (box.bottom < -height * 0.2 || box.top > height * 1.2) { return; }
      var progress = (height * 0.9 - box.top) / (height * 0.5 + box.height);
      var lit = Math.max(0, Math.min(1, progress)) * (item.words.length + SCRUB_SOFT);
      item.words.forEach(function (word, index) {
        word.style.opacity = SCRUB_LOW + (1 - SCRUB_LOW) * Math.max(0, Math.min(1, (lit - index) / SCRUB_SOFT));
      });
    });
  }

  /* Figures count up when seen. */
  function formatLike(final, value) {
    var match = final.match(/^([^\d]*)([\d,]*)(\.\d+)?(.*)$/);
    if (!match) { return final; }
    var decimals = match[3] ? match[3].length - 1 : 0;
    var parts = Math.abs(value).toFixed(decimals).split(".");
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
  NYC.onMount("[data-countup]", function (element) { whenSeen(element, function () { NYC.countUp(element); }); });

  /* A wave through the letters. */
  NYC.onMount("[data-wave]", function (element) {
    var text = element.textContent;
    if (element.children.length || !text) { return; }
    element.setAttribute("aria-label", text);
    element.textContent = "";
    Array.prototype.forEach.call(text, function (letter, index) {
      var span = document.createElement("span");
      span.className = "wl"; span.textContent = letter;
      span.style.setProperty("--i", index);
      span.setAttribute("aria-hidden", "true");
      element.appendChild(span);
    });
  });

  /* Hover and press, delegated so Dash re-renders never lose them. */
  document.addEventListener("pointermove", function (event) {
    if (!fine.matches || NYC.reducedMotion() || !event.target.closest) { return; }
    var spot = event.target.closest(".spot");
    if (spot) {
      var area = spot.getBoundingClientRect();
      spot.style.setProperty("--mx", (event.clientX - area.left) + "px");
      spot.style.setProperty("--my", (event.clientY - area.top) + "px");
    }
    var magnet = event.target.closest(".magnetic");
    if (magnet && !magnet.disabled) {
      var box = magnet.getBoundingClientRect();
      magnet.style.translate = ((event.clientX - box.left - box.width / 2) * MAGNET) + "px " + ((event.clientY - box.top - box.height / 2) * MAGNET) + "px";
    }
  }, { passive: true });
  document.addEventListener("pointerout", function (event) {
    var magnet = event.target.closest && event.target.closest(".magnetic");
    if (magnet && !magnet.contains(event.relatedTarget)) { magnet.style.translate = ""; }
  }, { passive: true });
  document.addEventListener("pointerdown", function (event) {
    var button = event.target.closest && event.target.closest(".button");
    if (!button || NYC.reducedMotion()) { return; }
    var box = button.getBoundingClientRect(), ripple = document.createElement("span");
    ripple.className = "ripple"; ripple.style.left = (event.clientX - box.left) + "px"; ripple.style.top = (event.clientY - box.top) + "px";
    ripple.setAttribute("aria-hidden", "true");
    ripple.addEventListener("animationend", function () { ripple.remove(); });
    button.appendChild(ripple);
  }, { passive: true });

  /* Reading position, section in view, ground tint. */
  var ticking = false, sections = [], links = [];
  function update() {
    ticking = false;
    var doc = document.documentElement, height = window.innerHeight, top = window.scrollY;
    var page = document.querySelector(".page");
    if (page) { if (top > 40) { page.setAttribute("data-scrolled", ""); } else { page.removeAttribute("data-scrolled"); } }
    var thread = document.getElementById("thread");
    if (thread) { thread.style.setProperty("--read", Math.min(1, top / Math.max(1, doc.scrollHeight - height)).toFixed(4)); }
    updateScrub(height);
    var current = null;
    sections.forEach(function (section) {
      var box = section.getBoundingClientRect();
      if (box.top < height * 0.45 && box.bottom > height * 0.45) { current = section; }
    });
    if (current) {
      var tint = TINTS[current.getAttribute("data-tint")] || TINTS.midnight;
      document.documentElement.style.setProperty("--tint", tint);
      links.forEach(function (link) {
        var mine = link.getAttribute("href") === "#" + current.id;
        if (mine) { link.setAttribute("aria-current", "true"); } else { link.removeAttribute("aria-current"); }
      });
    }
  }
  function onScroll() {
    if (ticking) { return; }
    ticking = true;
    window.requestAnimationFrame(update);
  }
  NYC.onMount("[data-tint]", function (section) { sections.push(section); onScroll(); });
  NYC.onMount(".nav-link", function (link) { links.push(link); });

  function start() {
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll, { passive: true });
    scan();
    onScroll();
  }
  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); } else { start(); }
})();
