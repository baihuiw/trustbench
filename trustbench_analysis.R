# ═══════════════════════════════════════════════════════════════
# TrustBench V2: Analysis & Visualization
# ═══════════════════════════════════════════════════════════════
# 
# Input: Three JSONL result files from run_v2.py
# Output: Publication-quality ggplot2 figures
#
# Usage:
#   1. Set the file paths in Section 0
#   2. Run the full script or source section by section
#   3. Figures saved to figures/ directory
# ═══════════════════════════════════════════════════════════════

library(jsonlite)
library(tidyverse)
library(patchwork)
library(scales)


files <- c(
  "outputs/results/v2_opus47_all.jsonl",
  "outputs/results/v2_gpt55_all.jsonl",
  "outputs/results/v2_gemini31_all.jsonl"
)

fig_dir <- "figures"
dir.create(fig_dir, showWarnings = FALSE)

# Model display names
model_labels <- c(
  "anthropic/claude-opus-4.7" = "Claude Opus 4.7",
  "openai/gpt-5.5" = "GPT-5.5",
  "google/gemini-3.1-pro-preview" = "Gemini 3.1 Pro"
)

# ── Section 1: Load and parse data ───────────────────────────

parse_jsonl <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nchar(trimws(lines)) > 0]
  
  map_dfr(lines, function(line) {
    j <- fromJSON(line, flatten = TRUE)
    tibble(
      prompt_id     = j$metadata$prompt_id,
      item_id       = j$metadata$item_id,
      section       = j$metadata$section,
      institution   = j$metadata$institution %||% "",
      condition     = j$metadata$condition,
      numbering     = j$metadata$numbering,
      framing       = j$metadata$framing,
      response_type = j$metadata$response_type,
      reverse_coded = j$metadata$reverse_coded,
      model         = j$response$model,
      repetition    = j$response$repetition,
      raw_text      = j$response$raw_text %||% "",
      choice        = j$response$canonical_choice %||% NA_character_,
      justification = j$response$justification %||% NA_character_,
      error         = j$response$error %||% NA_character_,
    )
  })
}

cat("Loading results...\n")
df <- map_dfr(files, function(f) {
  if (file.exists(f)) {
    cat(sprintf("  %s\n", f))
    parse_jsonl(f)
  } else {
    warning(sprintf("File not found: %s", f))
    tibble()
  }
})

# Clean up
df <- df %>%
  mutate(
    model_label = recode(model, !!!model_labels),
    choice_num  = as.numeric(choice),
    refused     = is.na(choice_num)
  )

cat(sprintf("\nLoaded %d rows across %d models\n", nrow(df), n_distinct(df$model)))

# Section labels for display
section_labels <- c(
  "wvs_confidence"     = "Institutional Confidence",
  "wvs_politicians"    = "Politician Trust",
  "wvs_social_general" = "Generalized Social Trust",
  "wvs_social_groups"  = "Social Group Trust",
  "social_role_trust"  = "Social Role Trust"
)

# ── Section 2: Refusal rate analysis ─────────────────────────

cat("\n═══ REFUSAL ANALYSIS ═══\n")
refusal_summary <- df %>%
  group_by(model_label, section, numbering, framing, response_type) %>%
  summarise(
    n = n(),
    n_refused = sum(refused),
    refusal_pct = 100 * mean(refused),
    .groups = "drop"
  ) %>%
  arrange(model_label, desc(refusal_pct))

# Print high refusal conditions
refusal_summary %>%
  filter(refusal_pct > 10) %>%
  print(n = 50)

# Refusal by model (overall)
refusal_by_model <- df %>%
  group_by(model_label) %>%
  summarise(
    n = n(),
    n_refused = sum(refused),
    refusal_pct = round(100 * mean(refused), 1),
    .groups = "drop"
  )
cat("\nOverall refusal rates:\n")
print(refusal_by_model)

# ── Figure 0: Refusal rates by model × condition ─────────────

p_refusal <- df %>%
  group_by(model_label, numbering, framing, response_type) %>%
  summarise(refusal_pct = 100 * mean(refused), .groups = "drop") %>%
  unite("condition", numbering, framing, response_type, sep = "\n") %>%
  ggplot(aes(x = condition, y = refusal_pct, fill = model_label)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  scale_fill_manual(values = c("GPT-5.5" = "#2b6cb0", "Claude Opus 4.7" = "#D85A30", "Gemini 3.1 Pro" = "#1D9E75")) +
  labs(
    title = "Refusal rates by model and condition",
    subtitle = "Percentage of responses where model did not provide a valid choice",
    x = NULL, y = "Refusal rate (%)", fill = "Model"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(size = 7, angle = 45, hjust = 1),
    legend.position = "top"
  )

ggsave(file.path(fig_dir, "fig0_refusal_rates.pdf"), p_refusal, width = 14, height = 7)
ggsave(file.path(fig_dir, "fig0_refusal_rates.png"), p_refusal, width = 14, height = 7, dpi = 300)
cat("Saved fig0_refusal_rates\n")

# ── Filter to valid responses for remaining analyses ─────────

df_valid <- df %>% filter(!refused)
cat(sprintf("\nValid responses: %d / %d (%.1f%%)\n", 
            nrow(df_valid), nrow(df), 100 * nrow(df_valid) / nrow(df)))

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 1: Institutional Trust Profile
# Primary condition: original numbering, survey framing, choice_justify
# ═══════════════════════════════════════════════════════════════

cat("\n═══ ANALYSIS 1: INSTITUTIONAL TRUST PROFILE ═══\n")

df_primary <- df_valid %>%
  filter(
    section == "wvs_confidence",
    numbering == "original",
    framing == "survey",
    response_type == "choice_justify"
  )

trust_profile <- df_primary %>%
  group_by(model_label, item_id, institution) %>%
  summarise(
    mean_trust = mean(choice_num, na.rm = TRUE),
    sd_trust   = sd(choice_num, na.rm = TRUE),
    n          = n(),
    se         = sd_trust / sqrt(n),
    .groups = "drop"
  )

# Order institutions by mean across all models
inst_order <- trust_profile %>%
  group_by(institution) %>%
  summarise(grand_mean = mean(mean_trust), .groups = "drop") %>%
  arrange(grand_mean) %>%
  pull(institution)

trust_profile <- trust_profile %>%
  mutate(institution = factor(institution, levels = inst_order))

p1 <- ggplot(trust_profile, aes(x = mean_trust, y = institution, color = model_label)) +
  geom_point(position = position_dodge(0.6), size = 2.5) +
  geom_errorbarh(
    aes(xmin = mean_trust - se, xmax = mean_trust + se),
    position = position_dodge(0.6), height = 0.3, linewidth = 0.4
  ) +
  scale_color_manual(values = c("GPT-4o" = "#2b6cb0", 
                                 "Claude Sonnet 4" = "#D85A30",
                                 "Gemini 2.5 Pro" = "#1D9E75")) +
  scale_x_continuous(
    limits = c(0.8, 4.2),
    breaks = 1:4,
    labels = c("1\nA great deal", "2\nQuite a lot", "3\nNot very much", "4\nNone at all")
  ) +
  labs(
    title = "Institutional trust profile by model",
    subtitle = "E[X] on 1–4 scale (lower = more trust) | Original numbering, survey framing, with justification",
    x = NULL, y = NULL, color = "Model"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    panel.grid.major.y = element_blank()
  )

ggsave(file.path(fig_dir, "fig1_trust_profile.pdf"), p1, width = 11, height = 9)
ggsave(file.path(fig_dir, "fig1_trust_profile.png"), p1, width = 11, height = 9, dpi = 300)
cat("Saved fig1_trust_profile\n")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 2: Politician/Government Trust (Q292)
# Diverging bar chart, reverse-coded items flipped
# ═══════════════════════════════════════════════════════════════

cat("\n═══ ANALYSIS 2: POLITICIAN TRUST ═══\n")

df_pol <- df_valid %>%
  filter(
    section == "wvs_politicians",
    numbering == "original",
    framing == "survey",
    response_type == "choice_justify"
  )

pol_labels <- tribble(
  ~item_id,   ~label,
  "trust_27", "Unsure whether to believe politicians",
  "trust_28", "Cautious about trusting politicians",
  "trust_29", "Politicians are open about decisions",
  "trust_30", "Government usually does the right thing",
  "trust_31", "Gov. information is unreliable",
  "trust_32", "Best to be cautious trusting gov.",
  "trust_33", "Politicians are honest and truthful",
  "trust_34", "Gov. people show poor judgement",
  "trust_35", "Politicians are incompetent",
  "trust_36", "Politicians put country above self",
  "trust_37", "Government has good intentions",
)

pol_profile <- df_pol %>%
  group_by(model_label, item_id) %>%
  summarise(
    raw_mean = mean(choice_num, na.rm = TRUE),
    reverse_coded = first(reverse_coded),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    # Align: higher = more distrust
    distrust = if_else(reverse_coded, raw_mean - 3.0, -(raw_mean - 3.0))
  ) %>%
  left_join(pol_labels, by = "item_id") %>%
  mutate(
    label_display = if_else(reverse_coded, paste0(label, " (R)"), label),
    item_type = if_else(reverse_coded, "Negative (R)", "Positive")
  )

# Order by mean distrust across models
pol_order <- pol_profile %>%
  group_by(label_display) %>%
  summarise(mean_distrust = mean(distrust), .groups = "drop") %>%
  arrange(mean_distrust) %>%
  pull(label_display)

pol_profile <- pol_profile %>%
  mutate(label_display = factor(label_display, levels = pol_order))

p2 <- ggplot(pol_profile, aes(x = distrust, y = label_display, fill = model_label)) +
  geom_col(position = position_dodge(0.7), width = 0.6) +
  geom_vline(xintercept = 0, linewidth = 0.8) +
  scale_fill_manual(values = c("GPT-4o" = "#2b6cb0", 
                                "Claude Sonnet 4" = "#D85A30",
                                "Gemini 2.5 Pro" = "#1D9E75")) +
  scale_x_continuous(
    limits = c(-2.5, 2.5),
    breaks = -2:2,
    labels = c("-2\n(Trusts)", "-1", "0\n(Neutral)", "+1", "+2\n(Distrusts)")
  ) +
  labs(
    title = "Politician & government distrust by model",
    subtitle = "Deviation from neutral (3.0) | Positive items flipped so right = distrust",
    x = NULL, y = NULL, fill = "Model"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    panel.grid.major.y = element_blank()
  )

ggsave(file.path(fig_dir, "fig2_politician_distrust.pdf"), p2, width = 11, height = 7)
ggsave(file.path(fig_dir, "fig2_politician_distrust.png"), p2, width = 11, height = 7, dpi = 300)
cat("Saved fig2_politician_distrust\n")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 3: Social Trust Gradient
# From personal (family) → social (strangers) → institutional → political
# ═══════════════════════════════════════════════════════════════

cat("\n═══ ANALYSIS 3: SOCIAL TRUST GRADIENT ═══\n")

# Combine all 4pt scale items across sections
df_4pt <- df_valid %>%
  filter(
    section %in% c("wvs_confidence", "wvs_social_groups", "social_role_trust"),
    numbering == "original",
    framing == "survey",
    response_type == "choice_justify"
  )

gradient_summary <- df_4pt %>%
  group_by(model_label, section, item_id, institution) %>%
  summarise(
    mean_trust = mean(choice_num, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

# Define trust target ordering (personal → institutional → political)
target_order <- c(
  # Social groups (Q58-Q63)
  "Your family", "Your neighborhood", "People you know personally",
  "People you meet for the first time", "People of another religion",
  "People of another nationality",
  # Social roles (custom)
  "A doctor", "A teacher", "Your neighbor", "A restaurant manager",
  "A stranger asking for directions", "People in general",
  # Top institutional (select)
  "Universities", "The armed forces", "Environmental organizations",
  "Charitable or humanitarian organizations",
  "The police", "The courts", "Banks", "The churches",
  # Low institutional
  "The government", "Parliament", "The press", "Television",
  "Political parties", "Major companies"
)

gradient_plot_data <- gradient_summary %>%
  filter(institution %in% target_order) %>%
  mutate(
    institution = factor(institution, levels = rev(target_order)),
    trust_category = case_when(
      section == "wvs_social_groups" ~ "Social groups",
      section == "social_role_trust" ~ "Social roles",
      section == "wvs_confidence" ~ "Institutions",
    )
  )

p3 <- ggplot(gradient_plot_data, aes(x = mean_trust, y = institution, color = model_label)) +
  geom_point(position = position_dodge(0.5), size = 2) +
  facet_grid(
    trust_category ~ ., scales = "free_y", space = "free_y",
    switch = "y"
  ) +
  scale_color_manual(values = c("GPT-4o" = "#2b6cb0", 
                                 "Claude Sonnet 4" = "#D85A30",
                                 "Gemini 2.5 Pro" = "#1D9E75")) +
  scale_x_continuous(limits = c(0.8, 4.2), breaks = 1:4) +
  labs(
    title = "Trust gradient: personal → social → institutional",
    subtitle = "E[X] on 1–4 scale (lower = more trust)",
    x = NULL, y = NULL, color = "Model"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    strip.placement = "outside",
    strip.text.y.left = element_text(angle = 0, face = "bold"),
    panel.grid.major.y = element_blank()
  )

ggsave(file.path(fig_dir, "fig3_trust_gradient.pdf"), p3, width = 11, height = 10)
ggsave(file.path(fig_dir, "fig3_trust_gradient.png"), p3, width = 11, height = 10, dpi = 300)
cat("Saved fig3_trust_gradient\n")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 4: Robustness — Numbering Effect
# ═══════════════════════════════════════════════════════════════

cat("\n═══ ANALYSIS 4: NUMBERING EFFECT ═══\n")

df_numbering <- df_valid %>%
  filter(section == "wvs_confidence") %>%
  group_by(model_label, item_id, institution, numbering) %>%
  summarise(mean_trust = mean(choice_num, na.rm = TRUE), n = n(), .groups = "drop")

# Pivot to wide for scatter
numbering_wide <- df_numbering %>%
  pivot_wider(names_from = numbering, values_from = c(mean_trust, n))

# Original vs Reversed
p4a <- ggplot(numbering_wide, aes(x = mean_trust_original, y = mean_trust_reversed, color = model_label)) +
  geom_point(size = 2, alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  facet_wrap(~ model_label) +
  scale_color_manual(values = c("GPT-4o" = "#2b6cb0", 
                                 "Claude Sonnet 4" = "#D85A30",
                                 "Gemini 2.5 Pro" = "#1D9E75")) +
  coord_equal(xlim = c(0.8, 4.2), ylim = c(0.8, 4.2)) +
  labs(
    title = "Numbering effect: Original (1→4) vs Reversed (1→4, labels flipped)",
    x = "Original E[X]", y = "Reversed E[X]"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none")

# Original vs Verbal
p4b <- ggplot(numbering_wide, aes(x = mean_trust_original, y = mean_trust_verbal, color = model_label)) +
  geom_point(size = 2, alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  facet_wrap(~ model_label) +
  scale_color_manual(values = c("GPT-4o" = "#2b6cb0", 
                                 "Claude Sonnet 4" = "#D85A30",
                                 "Gemini 2.5 Pro" = "#1D9E75")) +
  coord_equal(xlim = c(0.8, 4.2), ylim = c(0.8, 4.2)) +
  labs(
    title = "Numbering effect: Original (numbered) vs Verbal (no numbers)",
    x = "Original E[X]", y = "Verbal E[X]"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none")

p4_combined <- p4a / p4b
ggsave(file.path(fig_dir, "fig4_numbering_effect.pdf"), p4_combined, width = 12, height = 10)
ggsave(file.path(fig_dir, "fig4_numbering_effect.png"), p4_combined, width = 12, height = 10, dpi = 300)
cat("Saved fig4_numbering_effect\n")

# Print correlations
numbering_cors <- numbering_wide %>%
  group_by(model_label) %>%
  summarise(
    r_orig_rev   = cor(mean_trust_original, mean_trust_reversed, use = "complete"),
    r_orig_verb  = cor(mean_trust_original, mean_trust_verbal, use = "complete"),
    mae_orig_rev = mean(abs(mean_trust_original - mean_trust_reversed), na.rm = TRUE),
    mae_orig_verb = mean(abs(mean_trust_original - mean_trust_verbal), na.rm = TRUE),
    .groups = "drop"
  )
cat("\nNumbering effect correlations and MAE:\n")
print(numbering_cors)


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 7: Cross-Model Agreement
# ═══════════════════════════════════════════════════════════════

cat("\n═══ ANALYSIS 7: CROSS-MODEL AGREEMENT ═══\n")

# Use primary condition
model_comparison <- df_primary %>%
  group_by(model_label, item_id, institution) %>%
  summarise(mean_trust = mean(choice_num, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = model_label, values_from = mean_trust)

# Pairwise correlations
models_present <- intersect(names(model_comparison), c("GPT-4o", "Claude Sonnet 4", "Gemini 2.5 Pro"))
if (length(models_present) >= 2) {
  cat("\nPairwise correlations (institutional confidence items):\n")
  for (i in 1:(length(models_present) - 1)) {
    for (j in (i + 1):length(models_present)) {
      m1 <- models_present[i]
      m2 <- models_present[j]
      r <- cor(model_comparison[[m1]], model_comparison[[m2]], use = "complete")
      cat(sprintf("  %s vs %s: r = %.3f\n", m1, m2, r))
    }
  }

  # Scatter matrix
  if (length(models_present) == 3) {
    p7a <- ggplot(model_comparison, aes(x = .data[[models_present[1]]], y = .data[[models_present[2]]])) +
      geom_point(size = 2, alpha = 0.7, color = "#2b6cb0") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
      coord_equal(xlim = c(0.8, 4.2), ylim = c(0.8, 4.2)) +
      labs(x = models_present[1], y = models_present[2]) +
      theme_minimal()
    
    p7b <- ggplot(model_comparison, aes(x = .data[[models_present[1]]], y = .data[[models_present[3]]])) +
      geom_point(size = 2, alpha = 0.7, color = "#D85A30") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
      coord_equal(xlim = c(0.8, 4.2), ylim = c(0.8, 4.2)) +
      labs(x = models_present[1], y = models_present[3]) +
      theme_minimal()
    
    p7c <- ggplot(model_comparison, aes(x = .data[[models_present[2]]], y = .data[[models_present[3]]])) +
      geom_point(size = 2, alpha = 0.7, color = "#1D9E75") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
      coord_equal(xlim = c(0.8, 4.2), ylim = c(0.8, 4.2)) +
      labs(x = models_present[2], y = models_present[3]) +
      theme_minimal()
    
    p7 <- (p7a | p7b | p7c) +
      plot_annotation(
        title = "Cross-model agreement on institutional trust",
        subtitle = "Per-item E[X] comparison (primary condition)"
      )
    
    ggsave(file.path(fig_dir, "fig7_model_agreement.pdf"), p7, width = 14, height = 5)
    ggsave(file.path(fig_dir, "fig7_model_agreement.png"), p7, width = 14, height = 5, dpi = 300)
    cat("Saved fig7_model_agreement\n")
  }
}

# ═══════════════════════════════════════════════════════════════
# SUMMARY TABLE: Export for paper
# ═══════════════════════════════════════════════════════════════

cat("\n═══ SUMMARY TABLE ═══\n")

summary_table <- df_valid %>%
  filter(
    section == "wvs_confidence",
    numbering == "original",
    framing == "survey",
    response_type == "choice_justify"
  ) %>%
  group_by(model_label, institution) %>%
  summarise(
    mean = round(mean(choice_num, na.rm = TRUE), 2),
    sd = round(sd(choice_num, na.rm = TRUE), 2),
    n = n(),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = model_label,
    values_from = c(mean, sd, n),
    names_glue = "{model_label}_{.value}"
  )

write_csv(summary_table, file.path(fig_dir, "summary_institutional_trust.csv"))
cat("Saved summary_institutional_trust.csv\n")

cat("\n═══ ALL DONE ═══\n")
cat(sprintf("Figures saved to %s/\n", fig_dir))
