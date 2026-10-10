/* Two small fixes that Dash and Plotly do not offer as options, and two pieces of feedback.

   1. Names for the dropdowns. dcc.Dropdown renders a combobox that the <label for="..."> beside
      it cannot reach, so a screen reader would announce it without a name. Each combobox gets
      the id of its visible label through aria-labelledby. React removes the attribute when it
      re-renders the dropdown, so the observer also watches that attribute and puts it back.

   2. Page scrolling on touch screens. A Plotly chart captures a finger drag to draw a zoom box,
      which traps anyone scrolling down a page of charts on a phone. On coarse pointers the
      charts get dragmode: false, which lets the drag scroll the page (tapping a mark still
      shows its values). The map keeps its gestures, because a map is expected to pan and pinch.

   Both are re-applied after every re-render through one throttled MutationObserver.

   3. A new report. When #report stops loading, <body> carries data-report="fresh" for a moment;
      site.css uses it to settle the four headline numbers into place. Nothing is set when the
      reader asks for reduced motion.

   4. The section in view. The matching masthead link gets aria-current, which site.css draws
      as a rule under the link. */
(function () {
  "use strict";

  var REFRESH_DELAY_MS = 60;
  var coarsePointer = window.matchMedia("(pointer: coarse)").matches;

  function nameComboboxes() {
    document.querySelectorAll(".field .dash-dropdown").forEach(function (dropdown) {
      var label = document.getElementById(dropdown.id + "-label");
      var combobox = dropdown.querySelector("[role='combobox']");
      if (label && combobox && combobox.getAttribute("aria-labelledby") !== label.id) {
        combobox.setAttribute("aria-labelledby", label.id);
      }
    });
  }

  function freeTouchScroll() {
    if (!coarsePointer || !window.Plotly) { return; }
    document.querySelectorAll(".graph:not(.graph-map) .js-plotly-plot").forEach(function (plot) {
      if (plot._fullLayout && plot._fullLayout.dragmode !== false) {
        window.Plotly.relayout(plot, { dragmode: false });
      }
    });
  }

  var FRESH_MS = 700;                 // longer than the longest settle animation plus its delay
  var LOADING_CLASS = "is-loading";
  var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  var watchedReport = null;
  var freshTimer = 0;

  function watchReport() {
    var report = document.getElementById("report");
    if (!report || report === watchedReport) { return; }
    watchedReport = report;
    var wasLoading = report.classList.contains(LOADING_CLASS);
    new MutationObserver(function () {
      var isLoading = report.classList.contains(LOADING_CLASS);
      if (wasLoading && !isLoading && !reducedMotion.matches) {
        document.body.setAttribute("data-report", "fresh");
        window.clearTimeout(freshTimer);
        freshTimer = window.setTimeout(function () { document.body.removeAttribute("data-report"); }, FRESH_MS);
      }
      wasLoading = isLoading;
    }).observe(report, { attributes: true, attributeFilter: ["class"] });
  }

  var spying = false;
  function watchSections() {
    var links = document.querySelectorAll(".nav a[href^='#']");
    if (spying || !links.length || !("IntersectionObserver" in window)) { return; }
    spying = true;
    var visible = {};
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) { visible[entry.target.id] = entry.isIntersecting; });
      var current = null;                       // the first section, in page order, that crosses the reading line
      links.forEach(function (link) {
        var id = link.getAttribute("href").slice(1);
        if (current === null && visible[id]) { current = id; }
      });
      links.forEach(function (link) {
        if (link.getAttribute("href").slice(1) === current) {
          link.setAttribute("aria-current", "true");
        } else {
          link.removeAttribute("aria-current");
        }
      });
    }, { rootMargin: "-30% 0px -65% 0px" });    // a thin band about a third of the way down the window
    links.forEach(function (link) {
      var section = document.getElementById(link.getAttribute("href").slice(1));
      if (section) { observer.observe(section); }
    });
  }

  var scheduled = false;
  function refresh() {
    scheduled = false;
    nameComboboxes();
    freeTouchScroll();
    watchReport();
    watchSections();
  }

  function start() {
    new MutationObserver(function () {
      if (scheduled) { return; }
      scheduled = true;
      window.setTimeout(refresh, REFRESH_DELAY_MS);   // a timer, not rAF: rAF never fires in a hidden tab
    }).observe(document.body, { childList: true, subtree: true, attributes: true,
                                attributeFilter: ["aria-labelledby"] });
    refresh();
  }

  // Dash may add this file after the document has finished loading, so do not rely on the event alone.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
