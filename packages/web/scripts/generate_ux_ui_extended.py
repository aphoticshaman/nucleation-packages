#!/usr/bin/env python3
"""Extended UX/UI design and engineering training data for desktop/mobile intel platforms."""
import json

def generate_examples():
    examples = []

    # ==========================================================================
    # DASHBOARD COMPOSITION
    # ==========================================================================
    dashboard = [
        "How should intel dashboard hierarchy prioritize critical vs. routine information?",
        "What grid systems work best for data-dense intelligence dashboards?",
        "How do you balance information density with cognitive load on dashboards?",
        "What widget sizing principles work for mixed-priority intel content?",
        "How should dashboards adapt between overview and detail modes?",
        "What refresh patterns work for real-time intel dashboards?",
        "How do you design dashboard states for no-data vs. loading vs. error?",
        "What card-based layouts work for intel item organization?",
        "How should KPI tiles be sized and positioned for scanning?",
        "What sparkline and mini-chart patterns work in dashboard context?",
        "How do you design dashboard customization without overwhelming users?",
        "What drag-and-drop patterns work for dashboard personalization?",
        "How should saved dashboard views be organized and accessed?",
        "What print and export considerations affect dashboard design?",
        "How do you design dashboards that work at different viewport sizes?",
    ]
    for q in dashboard:
        examples.append({"instruction": q, "input": "", "output": "[Dashboard design: composition principles, information hierarchy, responsive adaptation, and user customization patterns.]"})

    # ==========================================================================
    # ALERT FATIGUE & NOTIFICATION DESIGN
    # ==========================================================================
    alerts = [
        "How do you prevent alert fatigue in high-volume intel systems?",
        "What notification priority schemes work for intelligence platforms?",
        "How should critical vs. informational alerts be visually differentiated?",
        "What sound design principles apply to intel alert systems?",
        "How do you design alert aggregation to reduce noise?",
        "What snooze and mute patterns work for alert management?",
        "How should alert thresholds be user-configurable?",
        "What digest modes work for batching non-urgent notifications?",
        "How do you design escalation for unacknowledged critical alerts?",
        "What haptic patterns work for mobile intel notifications?",
        "How should notification history be organized and searchable?",
        "What quiet hours and focus modes should intel apps support?",
        "How do you design alerts that interrupt appropriately by context?",
        "What machine learning can reduce notification noise?",
        "How should cross-device notification sync work?",
    ]
    for q in alerts:
        examples.append({"instruction": q, "input": "", "output": "[Notification design: priority classification, fatigue prevention, escalation patterns, and cross-device synchronization.]"})

    # ==========================================================================
    # INFORMATION DENSITY VS. CLARITY
    # ==========================================================================
    density = [
        "How do you maximize data density without sacrificing readability?",
        "What typography choices support dense information display?",
        "How do micro-interactions reduce cognitive load in dense UIs?",
        "What whitespace principles apply to data-heavy interfaces?",
        "How do you design tables that show many columns effectively?",
        "What progressive disclosure patterns work for layered detail?",
        "How should hover states reveal additional information?",
        "What tooltip patterns work for data point explanation?",
        "How do you design expandable rows in data tables?",
        "What truncation and overflow patterns work for text-heavy content?",
        "How should numbers and units be formatted for quick scanning?",
        "What abbreviation conventions work for space-constrained UIs?",
        "How do you design data labels that don't obscure data?",
        "What zoom and pan patterns work for dense visualizations?",
        "How do you balance data density between expert and novice users?",
    ]
    for q in density:
        examples.append({"instruction": q, "input": "", "output": "[Information density: typography optimization, progressive disclosure, micro-interactions, and expert vs novice considerations.]"})

    # ==========================================================================
    # MOBILE INTEL CONSUMPTION
    # ==========================================================================
    mobile = [
        "How should intel dashboards adapt to mobile screen constraints?",
        "What thumb-zone considerations affect mobile intel UI?",
        "How do you design one-handed operation for mobile intel apps?",
        "What swipe gestures work for intel item navigation?",
        "How should mobile alerts differ from desktop notifications?",
        "What offline modes should mobile intel apps support?",
        "How do you design map interactions for mobile intel?",
        "What pull-to-refresh patterns work for intel feeds?",
        "How should mobile intel apps handle connectivity changes?",
        "What bottom sheet patterns work for detail views on mobile?",
        "How do you design mobile search for intel queries?",
        "What share sheet integrations benefit intel workflows?",
        "How should mobile intel apps use device sensors?",
        "What widget patterns work for iOS and Android home screens?",
        "How do you design mobile onboarding for complex intel apps?",
    ]
    for q in mobile:
        examples.append({"instruction": q, "input": "", "output": "[Mobile design: screen adaptation, gesture patterns, offline capability, and platform-specific considerations.]"})

    # ==========================================================================
    # iOS SPECIFIC DESIGN
    # ==========================================================================
    ios = [
        "How do iOS Human Interface Guidelines apply to intel apps?",
        "What SF Symbols work for intel and security iconography?",
        "How should iOS intel apps use Dynamic Type for accessibility?",
        "What UIKit vs SwiftUI tradeoffs affect intel app development?",
        "How should intel apps use iOS Focus modes?",
        "What WidgetKit patterns work for intel at-a-glance views?",
        "How should intel apps integrate with iOS Shortcuts?",
        "What Live Activities patterns work for ongoing intel events?",
        "How should intel apps use iOS background refresh?",
        "What Handoff patterns work for iOS-Mac intel workflows?",
        "How should intel apps handle iPad multitasking and Stage Manager?",
        "What iOS accessibility features must intel apps support?",
        "How should intel apps use iOS location services appropriately?",
        "What iOS notification management features should intel apps support?",
        "How should intel apps design for different iPhone screen sizes?",
    ]
    for q in ios:
        examples.append({"instruction": q, "input": "", "output": "[iOS design: HIG compliance, system integration, accessibility, and platform-specific feature utilization.]"})

    # ==========================================================================
    # ANDROID SPECIFIC DESIGN
    # ==========================================================================
    android = [
        "How does Material Design 3 apply to intel applications?",
        "What Android adaptive icons work for intel app branding?",
        "How should Android intel apps handle different screen densities?",
        "What Jetpack Compose patterns work for intel interfaces?",
        "How should intel apps use Android notification channels?",
        "What widget patterns work for Android home screen intel?",
        "How should intel apps handle Android background restrictions?",
        "What work profile considerations affect enterprise intel apps?",
        "How should Android intel apps handle different API levels?",
        "What foldable and large screen considerations affect Android intel apps?",
        "How should Android intel apps integrate with Google Assistant?",
        "What Android accessibility services must intel apps support?",
        "How should intel apps use Android location in different modes?",
        "What Material You theming works for intel app personalization?",
        "How should Android intel apps handle manufacturer UI variations?",
    ]
    for q in android:
        examples.append({"instruction": q, "input": "", "output": "[Android design: Material Design compliance, device fragmentation, system integration, and platform-specific patterns.]"})

    # ==========================================================================
    # TABLET DESIGN
    # ==========================================================================
    tablet = [
        "How should intel interfaces use tablet screen real estate?",
        "What multi-column layouts work for tablet intel apps?",
        "How should navigation differ between tablet and phone intel apps?",
        "What split-view patterns work for master-detail intel display?",
        "How should intel apps handle tablet orientation changes?",
        "What stylus interactions enhance tablet intel workflows?",
        "How should tablet intel apps use floating windows?",
        "What keyboard shortcut support should tablet intel apps have?",
        "How should tablet intel apps integrate with external displays?",
        "What drag-and-drop patterns work for tablet intel apps?",
        "How should tablet intel apps handle pointer input?",
        "What pop-over patterns work for tablet secondary content?",
        "How should tablet intel apps adapt to different aspect ratios?",
        "What compact vs. regular size class considerations affect tablets?",
        "How should tablet intel apps support picture-in-picture?",
    ]
    for q in tablet:
        examples.append({"instruction": q, "input": "", "output": "[Tablet design: screen utilization, multi-pane layouts, input modalities, and orientation handling.]"})

    # ==========================================================================
    # ACCESSIBILITY FOR HIGH-STAKES DECISIONS
    # ==========================================================================
    accessibility = [
        "How do WCAG guidelines apply to mission-critical intel interfaces?",
        "What color contrast requirements apply to urgent intel displays?",
        "How should screen readers navigate complex intel dashboards?",
        "What keyboard navigation patterns work for intel applications?",
        "How should intel apps handle color-blind users viewing risk maps?",
        "What ARIA labels are needed for dynamic intel content?",
        "How should intel apps support reduced motion preferences?",
        "What focus management is needed for modal intel workflows?",
        "How should intel apps handle users with cognitive disabilities?",
        "What text scaling requirements affect intel data display?",
        "How should audio and video intel content be captioned?",
        "What touch target sizes are needed for stress conditions?",
        "How should intel apps support switch control users?",
        "What error messaging is needed for accessible intel forms?",
        "How should intel apps handle users under time pressure?",
    ]
    for q in accessibility:
        examples.append({"instruction": q, "input": "", "output": "[Accessibility design: WCAG compliance, assistive technology support, cognitive considerations, and stress-condition usability.]"})

    # ==========================================================================
    # COLOR THEORY FOR DATA VISUALIZATION
    # ==========================================================================
    color = [
        "What color palettes work for risk-level encoding in intel?",
        "How do you design color scales for continuous data visualization?",
        "What diverging color schemes work for positive/negative values?",
        "How should color encode categorical intel data?",
        "What color-blind safe palettes work for intel visualization?",
        "How do you handle color in dark mode intel interfaces?",
        "What sequential color schemes work for magnitude encoding?",
        "How should alert colors be chosen for maximum attention?",
        "What background colors support prolonged data viewing?",
        "How do you use color to create visual hierarchy in dashboards?",
        "What color semantics are culturally universal vs. specific?",
        "How should brand colors integrate with data visualization?",
        "What color opacity patterns work for overlapping data?",
        "How do you design color legends that don't obscure data?",
        "What color animation patterns draw attention appropriately?",
    ]
    for q in color:
        examples.append({"instruction": q, "input": "", "output": "[Color design: encoding schemes, accessibility requirements, dark mode adaptation, and cultural considerations.]"})

    # ==========================================================================
    # DATA VISUALIZATION PATTERNS
    # ==========================================================================
    dataviz = [
        "What chart types work best for time series intel data?",
        "How do you design interactive choropleth maps for risk visualization?",
        "What network graph layouts work for entity relationship display?",
        "How should treemaps be designed for hierarchical intel data?",
        "What scatter plot patterns work for multi-dimensional analysis?",
        "How do you design small multiples for comparative intel?",
        "What sankey diagrams work for flow and relationship data?",
        "How should heatmaps be designed for pattern detection?",
        "What bubble charts work for sized categorical comparison?",
        "How do you design radar charts for multi-factor assessment?",
        "What bullet charts work for goal tracking in intel?",
        "How should box plots communicate distribution to non-statisticians?",
        "What waterfall charts work for change attribution?",
        "How do you design interactive filtering for complex visualizations?",
        "What annotation patterns work for chart storytelling?",
    ]
    for q in dataviz:
        examples.append({"instruction": q, "input": "", "output": "[Data visualization: chart selection, interaction design, annotation patterns, and user comprehension optimization.]"})

    # ==========================================================================
    # QUANT/STATS UI FEATURES
    # ==========================================================================
    quant_ui = [
        "How should Bayesian probability updates be visualized for users?",
        "What UI shows confidence intervals and uncertainty effectively?",
        "How do you design Monte Carlo simulation result displays?",
        "What sliders and inputs work for scenario parameter adjustment?",
        "How should VaR and risk metrics be displayed to non-quants?",
        "What time series forecast UIs show uncertainty bands?",
        "How do you design correlation matrix visualizations?",
        "What regression result displays are interpretable by users?",
        "How should A/B test results be visualized for decision-making?",
        "What sensitivity analysis UIs help users understand drivers?",
        "How do you design distribution visualizations for user understanding?",
        "What threshold adjustment UIs work for signal detection?",
        "How should model performance metrics be displayed?",
        "What backtesting result UIs show strategy performance?",
        "How do you design what-if scenario comparison interfaces?",
    ]
    for q in quant_ui:
        examples.append({"instruction": q, "input": "", "output": "[Quantitative UI: statistical visualization, uncertainty display, parameter controls, and result interpretation aids.]"})

    # ==========================================================================
    # INTERACTIVE CONTROLS FOR ANALYSIS
    # ==========================================================================
    controls = [
        "What date range picker patterns work for temporal intel analysis?",
        "How should multi-select filters be designed for faceted search?",
        "What slider patterns work for continuous threshold adjustment?",
        "How do you design typeahead search for entity lookup?",
        "What toggle patterns work for layer visibility control?",
        "How should dropdown menus handle long option lists?",
        "What pill/chip patterns work for active filter display?",
        "How do you design comparison mode toggles for side-by-side?",
        "What zoom controls work for map and timeline navigation?",
        "How should save and share controls work for analysis states?",
        "What undo/redo patterns work for exploratory analysis?",
        "How do you design export options for different output formats?",
        "What keyboard shortcuts should analysis tools support?",
        "How should analysis history and breadcrumbs work?",
        "What query builder UIs work for advanced filtering?",
    ]
    for q in controls:
        examples.append({"instruction": q, "input": "", "output": "[Control design: filter patterns, selection mechanisms, state management, and advanced query interfaces.]"})

    # ==========================================================================
    # REAL-TIME AND STREAMING UI
    # ==========================================================================
    realtime = [
        "How should real-time data updates affect UI without jarring?",
        "What streaming indicator patterns show live data status?",
        "How do you design feeds that update without losing scroll position?",
        "What animation patterns indicate new data arrival?",
        "How should time-ago stamps update for streaming data?",
        "What buffering and catchup patterns work for real-time gaps?",
        "How do you design rate limiting indicators for fast streams?",
        "What reconnection patterns work for dropped connections?",
        "How should users control stream pause and resume?",
        "What highlighting patterns show changed values in updates?",
        "How do you design real-time charts that scroll or update in place?",
        "What audio cues work for background real-time alerts?",
        "How should real-time UIs handle high-frequency updates?",
        "What batching patterns reduce visual noise in streams?",
        "How do you design real-time vs. historical mode switching?",
    ]
    for q in realtime:
        examples.append({"instruction": q, "input": "", "output": "[Real-time design: update patterns, streaming indicators, scroll management, and connection state handling.]"})

    # ==========================================================================
    # EMPTY STATES AND BLANK MAP PHILOSOPHY
    # ==========================================================================
    empty = [
        "How do empty states guide users to their first action?",
        "What blank map designs invite exploration without overwhelming?",
        "How should zero-results states guide query refinement?",
        "What onboarding patterns work for complex intel tools?",
        "How do you design first-run experiences for intel platforms?",
        "What placeholder content communicates available functionality?",
        "How should sample data demonstrate tool capabilities?",
        "What progressive onboarding reveals features over time?",
        "How do you design contextual help that appears when needed?",
        "What walkthrough patterns work for complex workflows?",
        "How should tips and hints appear without annoying experts?",
        "What documentation integration works within the UI?",
        "How do you design feature discovery for power users?",
        "What empty state actions accelerate user activation?",
        "How should setup wizards guide initial configuration?",
    ]
    for q in empty:
        examples.append({"instruction": q, "input": "", "output": "[Empty state design: user guidance, exploration invitation, progressive revelation, and feature discovery patterns.]"})

    return examples

if __name__ == "__main__":
    examples = generate_examples()
    print(f"Generated {len(examples)} UX/UI examples")
    with open("ux_ui_extended_training.json", "w") as f:
        json.dump(examples, f, indent=2)
    print("Saved to ux_ui_extended_training.json")
