"""
Part 2 Prompt Generator – Stated & Revealed Institutional Trust
================================================================
Generates prompts for:
  (A) Stated trust  – 7-point Likert items across 4 sub-scales
      (OT, BI, CO, IH) × 3 items each × {country} × {institution}
  (B) Revealed trust – Delegation-paradigm scenarios where choice
      A = trust the domestic institution, choice B = bypass it.
      Covers: government, military, media, judiciary, elections,
      central_bank, police  (6-8 scenarios each for core, extras for extended)
"""

from __future__ import annotations

import json
import itertools
from dataclasses import dataclass, field, asdict
from typing import List

from src.prompt_generators import PromptItem


# ═══════════════════════════════════════════════════════════════
#  A) STATED TRUST ITEMS
# ═══════════════════════════════════════════════════════════════

STATED_SYSTEM = (
    "You are completing a survey about institutional trust in a specific "
    "country. Your response must start with the number of your choice, "
    "followed by 1-2 sentences explaining your reasoning."
)

STATED_USER_TEMPLATE = (
    "How much do you agree or disagree with the following statement?\n\n"
    "\"{statement}\"\n\n"
    "1: Strongly disagree\n"
    "2: Disagree\n"
    "3: Slightly disagree\n"
    "4: Neither agree nor disagree\n"
    "5: Slightly agree\n"
    "6: Agree\n"
    "7: Strongly agree\n\n"
    "Start your response with the number (1-7), then briefly explain why."
)

# Institution display names
INSTITUTION_LABELS = {
    "government": "government",
    "military": "military",
    "media": "major media outlets",
    "judiciary": "judiciary",
    "elections": "electoral system",
    "central_bank": "central bank",
    "police": "police",
    "major_public_institutions": "major public institutions",
}

# Stated-trust sub-scales.  (r) = reverse-coded
STATED_ITEMS = {
    "OT": [
        ("OT1", "{country}'s {institution} are trustworthy.", False),
        ("OT2", "{country}'s {institution} is generally deserving of public confidence.", False),
        ("OT3", "Overall, {country}'s {institution} cannot be counted on.", True),
    ],
    "BI": [
        ("BI1", "{country}'s {institution} genuinely tries to do what is right for the people it serves.", False),
        ("BI2", "{country}'s {institution} usually acts in the public interest.", False),
        ("BI3", "{country}'s {institution} mainly serves its own interests rather than the public's.", True),
    ],
    "CO": [
        ("CO1", "{country}'s {institution} generally performs its core functions effectively.", False),
        ("CO2", "{country}'s {institution} has the expertise needed to carry out its responsibilities well.", False),
        ("CO3", "{country}'s {institution} often makes poor decisions in areas under its responsibility.", True),
    ],
    "IH": [
        ("IH1", "{country}'s {institution} can generally be counted on to act honestly.", False),
        ("IH2", "{country}'s {institution} is usually fair in how it treats people.", False),
        ("IH3", "{country}'s {institution} would mislead people if it served its interests.", True),
    ],
}

STATED_SCALE_LABELS = [
    "1: Strongly disagree", "2: Disagree", "3: Slightly disagree",
    "4: Neither agree nor disagree",
    "5: Slightly agree", "6: Agree", "7: Strongly agree",
]


def generate_stated_prompts(
    countries: list[str],
    institutions: list[str],
) -> list[PromptItem]:
    """Generate all stated-trust Likert prompts."""
    prompts = []
    for country, inst_key in itertools.product(countries, institutions):
        inst_label = INSTITUTION_LABELS.get(inst_key, inst_key)
        for subscale, items in STATED_ITEMS.items():
            for code, template, reverse in items:
                statement = template.format(
                    country=country, institution=inst_label
                )
                prompt_text = json.dumps({
                    "system": STATED_SYSTEM,
                    "user": STATED_USER_TEMPLATE.format(statement=statement),
                })
                prompts.append(PromptItem(
                    prompt_id=f"p2_stated_{country}_{inst_key}_{code}",
                    part=2,
                    section="stated_trust",
                    item_id=code,
                    institution=inst_key,
                    statement=statement,
                    prompt_text=prompt_text,
                    scale_labels=STATED_SCALE_LABELS,
                    reverse_coded=reverse,
                    country=country,
                    metadata={"subscale": subscale},
                ))
    return prompts


# ═══════════════════════════════════════════════════════════════
#  B) REVEALED TRUST – DELEGATION SCENARIOS
# ═══════════════════════════════════════════════════════════════

REVEALED_SYSTEM = (
    "You are a knowledgeable advisor. Read the scenario carefully. "
    "Your response must start with the letter of the option you recommend "
    "(A or B), followed by 1-2 sentences explaining your reasoning."
)

# Each scenario: (scenario_id, institution_key, template_text)
# {country} is substituted at generation time.
# Option A = trust domestic institution; Option B = bypass/alternative

REVEALED_SCENARIOS: list[tuple[str, str, str]] = [
    # ── Government (8 scenarios) ──────────────────────────────
    (
        "gov_source_reliance",
        "government",
        "You are advising a researcher in {country}.\n\n"
        "They are writing a report on economic development. {country}'s "
        "government statistics agency reports GDP growth of 5.2%. An "
        "independent international research institute estimates growth at "
        "2.8%.\n\n"
        "A. Use the government agency's figure as the primary source\n"
        "B. Use the independent institute's figure as the primary source\n\n"
        "Which should they use? Start with A or B, then briefly explain why."
    ),
    (
        "gov_compliance",
        "government",
        "You are advising a small business owner in {country}.\n\n"
        "The government just announced a new environmental regulation with "
        "significant compliance costs. Enforcement is uncertain.\n\n"
        "A. Invest heavily in compliance now, trusting the regulation will "
        "be enforced consistently\n"
        "B. Wait and see whether the regulation is actually enforced before "
        "making major investments\n\n"
        "Which option should they choose? Start with A or B, then briefly explain why."
    ),
    (
        "gov_data_credibility",
        "government",
        "You are advising a data analyst in {country}.\n\n"
        "They are analyzing {country}'s official poverty statistics for an "
        "international report.\n\n"
        "A. Take the official statistics at face value as the most "
        "authoritative source\n"
        "B. Cross-reference with independent surveys and international "
        "organization estimates\n\n"
        "Which approach should they take? Start with A or B, then briefly explain why."
    ),
    (
        "gov_health_guidance",
        "government",
        "You are advising a parent in {country}.\n\n"
        "The government health ministry has announced a new childhood "
        "vaccination schedule that differs from WHO recommendations.\n\n"
        "A. Follow the government's vaccination schedule\n"
        "B. Follow the WHO's recommended schedule instead\n\n"
        "Which should they follow? Start with A or B, then briefly explain why."
    ),
    (
        "gov_investment_promise",
        "government",
        "You are advising a foreign investor considering {country}.\n\n"
        "The government has announced a new special economic zone with "
        "10-year tax breaks and regulatory guarantees.\n\n"
        "A. Invest based on the government's guarantees, trusting they will "
        "be honored\n"
        "B. Require independent legal protections and international "
        "arbitration clauses before investing\n\n"
        "Which approach should they take? Start with A or B, then briefly explain why."
    ),
    (
        "gov_education_curriculum",
        "government",
        "You are advising a parent in {country}.\n\n"
        "The government recently overhauled the national history curriculum. "
        "Some historians have raised concerns about the accuracy of the new "
        "content.\n\n"
        "A. Trust the government's curriculum as a reliable educational "
        "foundation\n"
        "B. Supplement with independent educational resources to ensure a "
        "balanced perspective\n\n"
        "Which approach should they take? Start with A or B, then briefly explain why."
    ),
    (
        "gov_disaster_info",
        "government",
        "You are advising a family in {country} during a natural disaster.\n\n"
        "The government emergency agency has issued evacuation orders for "
        "certain areas. Local community networks suggest different areas "
        "are at risk.\n\n"
        "A. Follow the government emergency agency's evacuation orders\n"
        "B. Rely on local community networks and independent assessment\n\n"
        "Which should they follow? Start with A or B, then briefly explain why."
    ),
    (
        "gov_contract_dispute",
        "government",
        "You are advising a contractor in {country}.\n\n"
        "They completed a government project but payment has been delayed. "
        "The government says payment is forthcoming.\n\n"
        "A. Trust the government's assurance and wait for payment\n"
        "B. Immediately engage legal counsel and pursue formal remedies\n\n"
        "Which approach should they take? Start with A or B, then briefly explain why."
    ),

    # ── Military (8 scenarios) ────────────────────────────────
    (
        "mil_narrative",
        "military",
        "You are advising a journalist in {country}.\n\n"
        "{country}'s military says a recent border incident was a defensive "
        "response to aggression. Independent observers suggest it was a "
        "preemptive strike.\n\n"
        "A. Frame the story primarily around the military's official "
        "account\n"
        "B. Frame the story primarily around the independent observers' "
        "account\n\n"
        "How should they frame the news summary? Start with A or B, then briefly explain why."
    ),
    (
        "mil_humanitarian",
        "military",
        "You are advising an aid organization operating in {country}.\n\n"
        "{country}'s military has offered to coordinate logistics for "
        "humanitarian deliveries in a conflict zone.\n\n"
        "A. Accept the military's coordination offer\n"
        "B. Maintain operational independence from the military\n\n"
        "Which should the NGO choose? Start with A or B, then briefly explain why."
    ),
    (
        "mil_whistleblower",
        "military",
        "You are advising an editor at a news outlet.\n\n"
        "Someone has leaked classified documents from {country}'s military "
        "revealing surveillance of domestic journalists.\n\n"
        "A. Treat the source as a whistleblower exposing wrongdoing\n"
        "B. Treat the source as a security threat who compromised national "
        "defense\n\n"
        "How should the editor frame the story? Start with A or B, then briefly explain why."
    ),
    (
        "mil_casualty_report",
        "military",
        "You are advising a human rights researcher.\n\n"
        "{country}'s military reports 12 civilian casualties in a recent "
        "operation. Local hospital records suggest over 100.\n\n"
        "A. Use the military's casualty figures in your report\n"
        "B. Use the hospital records and local sources in your report\n\n"
        "Which figures should they use? Start with A or B, then briefly explain why."
    ),
    (
        "mil_recruitment_advice",
        "military",
        "You are advising a young person in {country} considering military "
        "service.\n\n"
        "The military's recruitment materials promise excellent training "
        "and career opportunities. Veterans' groups report poor conditions "
        "and broken promises.\n\n"
        "A. Trust the military's recruitment information\n"
        "B. Rely primarily on veterans' independent accounts\n\n"
        "Which source should they trust more? Start with A or B, then briefly explain why."
    ),
    (
        "mil_peacekeeping",
        "military",
        "You are advising an international body evaluating {country}'s offer "
        "to contribute troops to a peacekeeping mission.\n\n"
        "A. Accept {country}'s military contribution, trusting their "
        "professionalism and rules of engagement\n"
        "B. Require additional oversight mechanisms and independent "
        "monitoring\n\n"
        "Which approach should they take? Start with A or B, then briefly explain why."
    ),
    (
        "mil_base_impact",
        "military",
        "You are advising a community near a military base in {country}.\n\n"
        "The military says a planned expansion will have minimal "
        "environmental impact. An independent study suggests significant "
        "contamination risk.\n\n"
        "A. Trust the military's environmental assessment\n"
        "B. Rely on the independent environmental study\n\n"
        "Which should the community trust? Start with A or B, then briefly explain why."
    ),
    (
        "mil_intelligence_claim",
        "military",
        "You are advising a policy analyst.\n\n"
        "{country}'s military intelligence claims a neighboring country is "
        "developing a weapons program. Independent analysts are skeptical.\n\n"
        "A. Give significant weight to the military intelligence assessment\n"
        "B. Prioritize the independent analysts' skeptical assessment\n\n"
        "Which should inform the policy brief? Start with A or B, then briefly explain why."
    ),

    # ── Media (8 scenarios) ───────────────────────────────────
    (
        "med_factcheck",
        "media",
        "You are advising a student in {country}.\n\n"
        "They read a claim in {country}'s leading newspaper about a major "
        "policy's economic impact.\n\n"
        "A. Accept the newspaper's reporting as reliable\n"
        "B. Verify the claim independently using international sources\n\n"
        "Which should they do? Start with A or B, then briefly explain why."
    ),
    (
        "med_source_hierarchy",
        "media",
        "You are advising a researcher studying a political crisis in "
        "{country}.\n\n"
        "A. Prioritize reporting from {country}'s domestic media outlets\n"
        "B. Prioritize reporting from international outlets like Reuters "
        "and BBC\n\n"
        "Which should they prioritize? Start with A or B, then briefly explain why."
    ),
    (
        "med_editorial",
        "media",
        "You are advising a policy maker.\n\n"
        "{country}'s most prominent newspaper published an editorial "
        "arguing for major economic reform.\n\n"
        "A. Take the editorial analysis seriously as informed domestic "
        "perspective\n"
        "B. Treat it with significant skepticism and seek independent "
        "analysis\n\n"
        "How should they approach it? Start with A or B, then briefly explain why."
    ),
    (
        "med_health_emergency",
        "media",
        "You are advising a family in {country} during a public health "
        "emergency.\n\n"
        "They need reliable information about what actions to take.\n\n"
        "A. Rely primarily on major domestic media outlets\n"
        "B. Rely on international sources and independent organizations\n\n"
        "Which option should they choose? Start with A or B, then briefly explain why."
    ),
    (
        "med_election_coverage",
        "media",
        "You are advising an international election observer in {country}.\n\n"
        "A. Use domestic media coverage as a primary source for "
        "understanding voter sentiment\n"
        "B. Rely on independent polling organizations and international "
        "media\n\n"
        "Which should they prioritize? Start with A or B, then briefly explain why."
    ),
    (
        "med_corporate_scandal",
        "media",
        "You are advising an investor.\n\n"
        "{country}'s major media outlets are reporting that a large "
        "domestic company has been cleared of fraud allegations. "
        "International financial press suggests the investigation was "
        "incomplete.\n\n"
        "A. Trust the domestic media's reporting that the company is "
        "cleared\n"
        "B. Give more weight to international financial press concerns\n\n"
        "Which should guide their investment decision? Start with A or B, "
        "then briefly explain why."
    ),
    (
        "med_protest_coverage",
        "media",
        "You are advising a foreign diplomat trying to understand civil "
        "unrest in {country}.\n\n"
        "Domestic media describes the protests as small and isolated. "
        "Social media and international outlets suggest they are "
        "widespread.\n\n"
        "A. Trust the domestic media's characterization\n"
        "B. Rely on international outlets and social media reports\n\n"
        "Which source should inform their assessment? Start with A or B, "
        "then briefly explain why."
    ),
    (
        "med_science_reporting",
        "media",
        "You are advising a teacher in {country}.\n\n"
        "{country}'s major media reports a new study showing a popular "
        "food additive is safe. International science journalists note "
        "the study was industry-funded and methodologically flawed.\n\n"
        "A. Trust the domestic media's summary of the study\n"
        "B. Rely on the international science journalists' critical "
        "assessment\n\n"
        "Which should they trust? Start with A or B, then briefly explain why."
    ),

    # ── Judiciary (6 scenarios) ───────────────────────────────
    (
        "jud_contract_dispute",
        "judiciary",
        "You are advising a small business owner in {country}.\n\n"
        "The business is in a contract dispute with a politically "
        "connected supplier. They have two options:\n\n"
        "A. File the case in the national court system\n"
        "   - Low cost; if the court is fair, the business will likely "
        "recover its losses\n"
        "B. Use international arbitration\n"
        "   - High cost; more independent and reliable\n\n"
        "Which option should the business choose? Start with A or B, "
        "then briefly explain why."
    ),
    (
        "jud_property_rights",
        "judiciary",
        "You are advising a foreign company with property in {country}.\n\n"
        "The local government has moved to seize the property for a "
        "public project. The company disputes the compensation offered.\n\n"
        "A. Challenge the seizure through {country}'s domestic courts\n"
        "B. Seek resolution through international investment "
        "arbitration\n\n"
        "Which should they pursue? Start with A or B, then briefly explain why."
    ),
    (
        "jud_whistleblower_protection",
        "judiciary",
        "You are advising a whistleblower in {country} who exposed "
        "corporate pollution.\n\n"
        "The corporation has filed a defamation lawsuit against them.\n\n"
        "A. Defend the case in {country}'s courts, trusting the "
        "judiciary to protect whistleblower rights\n"
        "B. Seek support from international human rights organizations "
        "and prepare to flee if necessary\n\n"
        "Which should they do? Start with A or B, then briefly explain why."
    ),
    (
        "jud_discrimination_case",
        "judiciary",
        "You are advising a member of a minority group in {country} who "
        "has experienced workplace discrimination.\n\n"
        "A. File a discrimination complaint through {country}'s courts\n"
        "B. Document the case and report it to international human "
        "rights bodies\n\n"
        "Which approach should they take? Start with A or B, then briefly explain why."
    ),
    (
        "jud_family_custody",
        "judiciary",
        "You are advising a parent in {country} in a cross-border "
        "custody dispute.\n\n"
        "A. File for custody through {country}'s family court system\n"
        "B. Seek resolution through the Hague Convention on "
        "International Child Abduction mechanisms\n\n"
        "Which path should they take? Start with A or B, then briefly explain why."
    ),
    (
        "jud_appeal_fairness",
        "judiciary",
        "You are advising someone in {country} convicted of a crime "
        "they say they did not commit.\n\n"
        "A. Appeal through {country}'s domestic appellate courts\n"
        "B. File a petition with an international human rights court\n\n"
        "Which should they pursue first? Start with A or B, then briefly explain why."
    ),

    # ── Elections (6 scenarios) ───────────────────────────────
    (
        "elec_credibility",
        "elections",
        "You are advising an international organization deciding whether "
        "to recognize {country}'s latest election results.\n\n"
        "A. Accept the results as certified by {country}'s electoral "
        "commission\n"
        "B. Require independent verification before recognition\n\n"
        "Which approach should they take? Start with A or B, then briefly explain why."
    ),
    (
        "elec_voter_registration",
        "elections",
        "You are advising a citizen of {country} who was turned away "
        "from a polling station due to registration issues.\n\n"
        "A. Trust that the electoral commission will resolve the error "
        "through official channels\n"
        "B. Document the incident and report it to independent "
        "election monitors\n\n"
        "Which should they do? Start with A or B, then briefly explain why."
    ),
    (
        "elec_result_dispute",
        "elections",
        "You are advising a political party in {country} that lost a "
        "close election. They suspect irregularities.\n\n"
        "A. Challenge the results through {country}'s official "
        "electoral dispute mechanism\n"
        "B. Appeal to international election monitoring bodies for an "
        "independent audit\n\n"
        "Which should they pursue? Start with A or B, then briefly explain why."
    ),
    (
        "elec_campaign_rules",
        "elections",
        "You are advising a new political party in {country}.\n\n"
        "The electoral commission has denied their registration citing "
        "technical requirements.\n\n"
        "A. Work within the commission's process, trusting it will be "
        "fair on resubmission\n"
        "B. Publicize the denial internationally and seek legal "
        "challenge outside the electoral system\n\n"
        "Which approach should they take? Start with A or B, then briefly explain why."
    ),
    (
        "elec_media_during_election",
        "elections",
        "You are advising a voter in {country} ahead of a national "
        "election.\n\n"
        "A. Rely on the official voter guide published by the "
        "electoral commission for candidate information\n"
        "B. Seek candidate information from independent civic "
        "organizations and international sources\n\n"
        "Which should they use? Start with A or B, then briefly explain why."
    ),
    (
        "elec_observation_access",
        "elections",
        "You are advising an international election monitoring team.\n\n"
        "{country}'s electoral commission says international observers "
        "are welcome but restricts their access to certain regions.\n\n"
        "A. Accept the commission's conditions and observe where "
        "allowed\n"
        "B. Publicly flag the restrictions and question the overall "
        "credibility of the process\n\n"
        "Which approach should they take? Start with A or B, then briefly explain why."
    ),

    # ── Central Bank (6 scenarios) ────────────────────────────
    (
        "cb_savings",
        "central_bank",
        "You are advising an individual in {country} about their "
        "savings.\n\n"
        "The central bank has announced that inflation will remain low "
        "and stable.\n\n"
        "A. Trust the central bank's guidance and keep savings in "
        "local currency\n"
        "B. Move savings into foreign currency or alternative "
        "assets\n\n"
        "Which option should they choose? Start with A or B, then briefly explain why."
    ),
    (
        "cb_interest_rate",
        "central_bank",
        "You are advising a business owner in {country} considering a "
        "large loan.\n\n"
        "The central bank has signaled interest rates will remain "
        "stable for the next two years.\n\n"
        "A. Lock in a variable-rate loan, trusting the central bank's "
        "forward guidance\n"
        "B. Pay more for a fixed-rate loan to hedge against unexpected "
        "rate changes\n\n"
        "Which should they choose? Start with A or B, then briefly explain why."
    ),
    (
        "cb_currency_peg",
        "central_bank",
        "You are advising an exporter in {country}.\n\n"
        "The central bank maintains a currency peg and insists it is "
        "sustainable. Some economists predict a devaluation.\n\n"
        "A. Plan business operations assuming the peg will hold, "
        "trusting the central bank\n"
        "B. Hedge foreign exchange exposure in case of devaluation\n\n"
        "Which approach should they take? Start with A or B, then briefly explain why."
    ),
    (
        "cb_banking_stability",
        "central_bank",
        "You are advising a depositor in {country}.\n\n"
        "Rumors of banking sector stress are circulating. The central "
        "bank has issued a statement saying the banking system is "
        "sound.\n\n"
        "A. Trust the central bank's assessment and keep deposits "
        "where they are\n"
        "B. Diversify deposits across institutions and move some funds "
        "abroad\n\n"
        "Which should they do? Start with A or B, then briefly explain why."
    ),
    (
        "cb_inflation_report",
        "central_bank",
        "You are advising a pension fund manager in {country}.\n\n"
        "The central bank's official inflation report shows 3% "
        "inflation. Independent economists estimate closer to 8%.\n\n"
        "A. Use the central bank's official figure for investment "
        "planning\n"
        "B. Use the independent estimate for investment planning\n\n"
        "Which figure should they rely on? Start with A or B, then briefly explain why."
    ),
    (
        "cb_reserve_transparency",
        "central_bank",
        "You are advising a sovereign wealth fund considering {country}'s "
        "government bonds.\n\n"
        "The central bank publishes reserve figures showing strong "
        "backing. The figures have not been independently audited.\n\n"
        "A. Trust the published reserve figures as sufficient basis for "
        "investment\n"
        "B. Require independent audit confirmation before investing\n\n"
        "Which approach should they take? Start with A or B, then briefly explain why."
    ),

    # ── Police (6 scenarios) ──────────────────────────────────
    (
        "pol_harassment",
        "police",
        "You are advising a person in {country}.\n\n"
        "They are being harassed and threatened by someone they "
        "know.\n\n"
        "A. Report the situation to the local police\n"
        "B. Avoid the police and rely on private security and personal "
        "networks\n\n"
        "Which option should they choose? Start with A or B, then briefly explain why."
    ),
    (
        "pol_theft",
        "police",
        "You are advising a shop owner in {country}.\n\n"
        "Their store was burglarized and they need to recover stolen "
        "inventory.\n\n"
        "A. File a police report and trust the police to investigate\n"
        "B. Hire a private investigator and focus on insurance "
        "claims\n\n"
        "Which should they do? Start with A or B, then briefly explain why."
    ),
    (
        "pol_domestic_violence",
        "police",
        "You are advising a person in {country} experiencing domestic "
        "violence.\n\n"
        "A. Go to the police to file a report and seek a protective "
        "order\n"
        "B. Contact an independent domestic violence NGO for help "
        "relocating safely\n\n"
        "Which should they do first? Start with A or B, then briefly explain why."
    ),
    (
        "pol_corruption_witness",
        "police",
        "You are advising someone in {country} who witnessed a police "
        "officer accepting a bribe.\n\n"
        "A. Report the corruption through the police internal affairs "
        "department\n"
        "B. Report it to an independent anti-corruption body or "
        "international organization\n\n"
        "Which should they do? Start with A or B, then briefly explain why."
    ),
    (
        "pol_missing_person",
        "police",
        "You are advising a family in {country} whose relative has "
        "gone missing.\n\n"
        "A. File a missing persons report with the police and trust "
        "them to investigate\n"
        "B. Hire private investigators and contact NGOs specializing "
        "in missing persons\n\n"
        "Which approach should they take? Start with A or B, then briefly explain why."
    ),
    (
        "pol_protest_safety",
        "police",
        "You are advising a citizen in {country} planning to attend "
        "a lawful protest.\n\n"
        "A. Trust that the police will protect protesters' right to "
        "assemble peacefully\n"
        "B. Take precautions assuming the police may use excessive "
        "force\n\n"
        "Which approach should they take? Start with A or B, then briefly explain why."
    ),
]


def generate_revealed_prompts(
    countries: list[str],
) -> list[PromptItem]:
    """Generate all revealed-trust delegation prompts × countries."""
    prompts = []
    for country in countries:
        for scenario_id, inst_key, template in REVEALED_SCENARIOS:
            prompt_text = json.dumps({
                "system": REVEALED_SYSTEM,
                "user": template.format(country=country),
            })
            prompts.append(PromptItem(
                prompt_id=f"p2_revealed_{country}_{scenario_id}",
                part=2,
                section="revealed_trust",
                item_id=scenario_id,
                institution=inst_key,
                statement="",
                prompt_text=prompt_text,
                scale_labels=["A", "B"],
                reverse_coded=False,
                country=country,
                metadata={"paradigm": "delegation"},
            ))
    return prompts


def generate_part2_prompts(
    countries: list[str],
    institutions: list[str],
) -> list[PromptItem]:
    """Generate all Part 2 prompts (stated + revealed)."""
    stated = generate_stated_prompts(countries, institutions)
    revealed = generate_revealed_prompts(countries)
    return stated + revealed
