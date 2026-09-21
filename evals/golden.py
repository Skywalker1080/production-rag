"""Golden eval set: Jio Financial Services annual report Q&A.

type=answer  -> scored by RAGAS (needs ground_truth + gold_pages for review)
type=refuse  -> rule-checked (answer must state the info is absent / decline)
"""
GOLDEN = [
    {
        "id": "rev-consol",
        "type": "answer",
        "question": "What was the consolidated revenue from operations for FY2025 and FY2026?",
        "ground_truth": "Consolidated revenue from operations was Rs 3,513.26 crore for FY2026 (year ended 31st March 2026) and Rs 2,042.91 crore for FY2025.",
        "gold_pages": [110],
    },
    {
        "id": "pat-standalone",
        "type": "answer",
        "question": "What was the standalone profit after tax for FY2026 and FY2025?",
        "ground_truth": "Standalone profit after tax was Rs 681.03 crore for FY2026 and Rs 548.91 crore for FY2025.",
        "gold_pages": [70],
    },
    {
        "id": "dividend-fy26",
        "type": "answer",
        "question": "What was the consolidated dividend income for FY2026?",
        "ground_truth": "Consolidated dividend income for FY2026 was Rs 268.97 crore (note 22).",
        "gold_pages": [110],
    },
    {
        "id": "finance-costs",
        "type": "answer",
        "question": "What were the consolidated finance costs for FY2026?",
        "ground_truth": "Consolidated finance costs for FY2026 were Rs 745.09 crore (note 26).",
        "gold_pages": [110],
    },
    {
        "id": "total-expenses",
        "type": "answer",
        "question": "What were the total consolidated expenses for FY2026?",
        "ground_truth": "Total consolidated expenses for FY2026 were Rs 1,982.93 crore.",
        "gold_pages": [110],
    },
    {
        "id": "eps-before",
        "type": "answer",
        "question": "What was the basic and diluted earnings per share before exceptional items for FY2026?",
        "ground_truth": "Basic and diluted EPS before exceptional items for FY2026 was Rs 2.41 (face value Rs 10 per share).",
        "gold_pages": [110],
    },
    {
        "id": "service-revenue-policy",
        "type": "answer",
        "question": "How does the company recognize service revenue including aggregator fees and BOU commission?",
        "ground_truth": "Service revenue including fees from aggregator and commission on BOU transactions is recognized on completion of provision of services, under Ind AS 115.",
        "gold_pages": [114],
    },
    {
        "id": "segment-revenue",
        "type": "answer",
        "question": "How is segment revenue identified and evaluated by the company?",
        "ground_truth": "Segment revenue, expenses, assets and liabilities are identified based on profit or loss evaluated for operating segments.",
        "gold_pages": [87],
    },
    {
        "id": "refuse-food",
        "type": "refuse",
        "question": "What is the CEO's favorite food?",
        "ground_truth": "The retrieved excerpts do not contain this information.",
        "gold_pages": [],
    },
    {
        "id": "refuse-poem",
        "type": "refuse",
        "question": "Write a poem about the monsoon.",
        "ground_truth": "I can only answer questions about company annual reports and filings.",
        "gold_pages": [],
    },
]
