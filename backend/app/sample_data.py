from typing import List
from app.schemas import SamplePrompt

SAMPLE_PROMPTS: List[SamplePrompt] = [
    SamplePrompt(
        id="varanasi-weaver",
        title="Varanasi Silk Weaver (Bhojpuri/Hindi)",
        dialect="bhojpuri",
        category="Handloom Textile",
        text="""Pranam! Humar naam Gauri Devi ba. Hum Varanasi Handloom Cluster me pichhle 18 years se authentic Banarasi brocade aur pure silk saree bunat baani. Contact number 9876543210 ba aur humar security pin 1234 rakhla ba. Humaar government Pehchan ID PEH-IND-88320 ha aur TRIFED registration number TRIFED-UP-VNS-1049 ba. Hum Banarasi Brocade GI tagged saari banawat baani."""
    ),
    SamplePrompt(
        id="channapatna-toys",
        title="Channapatna Wooden Toy Crafter (Kannada/English)",
        dialect="kannada",
        category="Wooden Craft & Toys",
        text="""Namaskara, I am Ramesh Gowda, a traditional lacquerware artisan from Channapatna Craft Cluster, Ramanagara, Karnataka. I have 12 years of experience crafting eco-friendly Channapatna wooden toys with natural vegetable dyes. Reach me at phone: 9123456780, pin: 4321. My Pehchan card is PEH-KA-44912 and TRIFED ID is TRIFED-KA-CHN-8821. We have GI certification for Channapatna Toys and Dolls."""
    ),
    SamplePrompt(
        id="madhubani-painter",
        title="Mithila Madhubani Folk Artist (Maithili/Hindi)",
        dialect="maithili",
        category="Folk Art & Painting",
        text="""Pranam, I am Sunita Jha from Madhubani, Bihar. Working under Mithila Painting Cluster for over 15 years creating Kohbar and Kachni style Madhubani handmade paintings on tussar silk and handmade paper. My mobile number is 9845123456, pin is 5678. Pehchan ID PEH-BR-99012 and TRIFED ID TRIFED-BR-MTH-3021. Our art is officially certified with Madhubani Paintings GI Tag."""
    ),
    SamplePrompt(
        id="kutch-embroidery",
        title="Kutch Embroidery Artisan (Gujarati/Hindi)",
        dialect="gujarati",
        category="Embroidery & Needlework",
        text="""Namaste! My name is Fatima Khatri from Bhuj, Kutch district in Gujarat. I have 20 years of expertise in authentic Kutch mirror work and Rogan art. Phone number: 9988776655, pin: 9876. Associated with Kutch Embroidery & Rogan Cluster, Pehchan ID PEH-GJ-77123, TRIFED-GJ-KCH-1102. Verified under Kutch Embroidery GI certification."""
    ),
]
