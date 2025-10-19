from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import google.generativeai as genai
import json
from datetime import datetime
import re
from fastapi import HTTPException

# Expanded Roadmap categories to cover wider industries
ROADMAP_CATEGORIES = {
    "role_based": [
        # IT Roles (original)
        "Frontend", "Backend", "Full Stack", "DevOps", "Data Analyst",
        "AI Engineer", "AI and Data Scientist", "Data Engineer", "Android",
        "Machine Learning", "PostgreSQL", "iOS", "Blockchain", "QA",
        "Software Architect", "Cyber Security", "UX Design", "Technical Writer",
        "Game Developer", "Server Side Game Developer", "MLOps", "Product Manager",
        "Engineering Manager", "Developer Relations", "BI Analyst",
        # Medical Roles
        "Physician", "Surgeon", "Nurse", "Pharmacist", "Dentist", "Physical Therapist",
        "Radiologist", "Veterinarian", "Medical Researcher", "Hospital Administrator",
        # Hospitality & Food Roles
        "Hotel Manager", "Restaurant Manager", "Chef", "Baker", "Event Planner",
        "Tour Guide", "Sommelier", "Nutritionist", "Food Scientist", "Barista",
        # Education Roles
        "Teacher", "Professor", "School Counselor", "Educational Administrator", "Tutor",
        # Legal Roles
        "Lawyer", "Paralegal", "Judge", "Legal Consultant",
        # Finance Roles
        "Accountant", "Financial Analyst", "Investment Banker", "Auditor", "Tax Advisor",
        # Engineering Roles
        "Mechanical Engineer", "Civil Engineer", "Electrical Engineer", "Chemical Engineer",
        "Aerospace Engineer",
        # Arts & Media Roles
        "Graphic Designer", "Musician", "Actor", "Journalist", "Photographer", "Film Director",
        # Business Roles
        "Marketing Specialist", "Sales Representative", "Human Resources Manager", "Entrepreneur",
        "Supply Chain Manager",
        # Other Roles
        "Pilot", "Architect", "Electrician", "Plumber", "Farmer", "Environmental Scientist"
    ],
    "skill_based": [
        # IT Skills (original)
        "SQL", "Computer Science", "React", "Vue", "Angular", "JavaScript",
        "TypeScript", "Node.js", "Python", "System Design", "Java", "ASP.NET Core",
        "API Design", "Spring Boot", "Flutter", "C++", "Rust", "Go",
        "Design and Architecture", "GraphQL", "React Native", "Design System",
        "Prompt Engineering", "MongoDB", "Linux", "Kubernetes", "Docker", "AWS",
        "Terraform", "Data Structures & Algorithms", "Redis", "Git and GitHub",
        "PHP", "Cloudflare", "AI Red Teaming", "AI Agents", "Next.js",
        "Code Review", "Kotlin", "HTML", "CSS",
        # Medical Skills
        "Anatomy", "Physiology", "Medical Diagnostics", "Surgical Techniques", "Pharmacology",
        "Patient Care", "Radiology Imaging", "Veterinary Medicine",
        # Hospitality & Food Skills
        "Cooking Techniques", "Menu Planning", "Hospitality Management", "Customer Service",
        "Event Coordination", "Wine Tasting", "Nutrition Planning", "Food Safety",
        "Baking Methods", "Barista Skills",
        # Education Skills
        "Teaching Methods", "Curriculum Design", "Classroom Management", "Educational Psychology",
        # Legal Skills
        "Legal Research", "Contract Law", "Courtroom Procedures", "Intellectual Property",
        # Finance Skills
        "Financial Accounting", "Investment Analysis", "Tax Preparation", "Auditing",
        # Engineering Skills
        "Mechanical Design", "Structural Analysis", "Circuit Design", "Chemical Processes",
        "Aerodynamics",
        # Arts & Media Skills
        "Graphic Design Software", "Music Composition", "Acting Techniques", "Journalism Writing",
        "Photography Editing", "Film Production",
        # Business Skills
        "Digital Marketing", "Sales Strategies", "HR Management", "Business Planning",
        "Supply Chain Logistics",
        # Other Skills
        "Aviation Navigation", "Architectural Drawing", "Electrical Wiring", "Plumbing Systems",
        "Agricultural Techniques", "Environmental Analysis"
    ]
}

# Pydantic models
class RoadmapRequest(BaseModel):
    domain: str
    category: str
    skill_level: Optional[str] = "beginner"
    custom_preferences: Optional[str] = None

class RoadmapNode(BaseModel):
    id: str
    title: str
    description: str
    level: int
    estimated_time_hours: int
    prerequisites: Optional[List[str]] = []
    key_topics: Optional[List[str]] = []
    resources: Optional[List[Dict[str, str]]] = []

class RoadmapConnection(BaseModel):
    from_node: str
    to_node: str
    relationship: Optional[str] = "leads_to"

class RoadmapResponse(BaseModel):
    domain: str
    category: str
    skill_level: str
    title: str
    description: str
    total_estimated_hours: int
    nodes: List[RoadmapNode]
    connections: List[RoadmapConnection]
    generated_at: str

def generate_roadmap_endpoint(request: RoadmapRequest):
    """Generate a personalized learning roadmap"""
    try:
        # Create detailed prompt for Gemini
        prompt = f"""
You are an expert education curriculum designer. Create a comprehensive, practical learning roadmap for: **{request.domain}**

**Context:**
- Category: {request.category}
- Current Skill Level: {request.skill_level}
{f'- Custom Preferences: {request.custom_preferences}' if request.custom_preferences else ''}

**Requirements:**
1. Create 10-15 progressive learning nodes that build upon each other
2. Each node must be a distinct, focused learning milestone
3. Include realistic time estimates (in hours) for each node
4. Show clear prerequisites and dependencies
5. Include 3-5 key topics/skills for each node
6. Suggest 2-3 high-quality learning resources per node (with real URLs when possible)
7. Order nodes from foundational to advanced topics

**CRITICAL: You MUST respond with ONLY valid JSON. No markdown, no code blocks, no extra text.**

Return this exact JSON structure:
{{
    "title": "Complete roadmap title (e.g., 'Frontend Developer Learning Path')",
    "description": "2-3 sentence overview of what this roadmap covers and career outcomes",
    "total_estimated_hours": 500,
    "nodes": [
        {{
            "id": "node_1",
            "title": "Short, clear topic name (2-5 words)",
            "description": "What you'll learn and why it matters (1-2 sentences)",
            "level": 1,
            "estimated_time_hours": 40,
            "prerequisites": [],
            "key_topics": ["Topic 1", "Topic 2", "Topic 3"],
            "resources": [
                {{
                    "title": "Resource name",
                    "url": "https://example.com",
                    "type": "course/tutorial/documentation/video",
                    "description": "Brief description"
                }}
            ]
        }},
        {{
            "id": "node_2",
            "title": "Next topic",
            "description": "Description",
            "level": 2,
            "estimated_time_hours": 50,
            "prerequisites": ["node_1"],
            "key_topics": ["Topic 1", "Topic 2", "Topic 3"],
            "resources": [
                {{
                    "title": "Resource name",
                    "url": "https://example.com",
                    "type": "course",
                    "description": "Brief description"
                }}
            ]
        }}
    ],
    "connections": [
        {{"from_node": "node_1", "to_node": "node_2", "relationship": "leads_to"}},
        {{"from_node": "node_2", "to_node": "node_3", "relationship": "leads_to"}}
    ]
}}

Make it industry-relevant, practical, and tailored to {request.skill_level} level.
Include real, working URLs for popular learning platforms (Udemy, Coursera, freeCodeCamp, MDN, official docs, etc.).
"""

        # Generate content with Gemini
        response = genai.GenerativeModel('gemini-2.0-flash').generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response text - remove markdown code blocks
        response_text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'^```\s*', '', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'\s*```$', '', response_text, flags=re.MULTILINE)
        response_text = response_text.strip()
        
        # Parse JSON
        try:
            roadmap_data = json.loads(response_text)
        except json.JSONDecodeError as json_err:
            # Try to extract JSON from text
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                roadmap_data = json.loads(json_match.group())
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse AI response as JSON: {str(json_err)}"
                )
        
        # Validate and construct response
        nodes = []
        for node_data in roadmap_data.get("nodes", []):
            nodes.append(RoadmapNode(
                id=node_data.get("id", f"node_{len(nodes) + 1}"),
                title=node_data.get("title", "Untitled Node"),
                description=node_data.get("description", ""),
                level=node_data.get("level", 1),
                estimated_time_hours=node_data.get("estimated_time_hours", 20),
                prerequisites=node_data.get("prerequisites", []),
                key_topics=node_data.get("key_topics", []),
                resources=node_data.get("resources", [])
            ))
        
        connections = []
        for conn_data in roadmap_data.get("connections", []):
            connections.append(RoadmapConnection(
                from_node=conn_data.get("from_node", ""),
                to_node=conn_data.get("to_node", ""),
                relationship=conn_data.get("relationship", "leads_to")
            ))
        
        return {
            "domain": request.domain,
            "category": request.category,
            "skill_level": request.skill_level,
            "title": roadmap_data.get("title", f"{request.domain} Learning Roadmap"),
            "description": roadmap_data.get("description", "A comprehensive learning path"),
            "total_estimated_hours": roadmap_data.get("total_estimated_hours", sum(n.estimated_time_hours for n in nodes)),
            "nodes": nodes,
            "connections": connections,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating roadmap: {str(e)}"
        )

def get_categories_endpoint():
    """Get all available roadmap categories"""
    return {
        "categories": ROADMAP_CATEGORIES,
        "total_role_based": len(ROADMAP_CATEGORIES["role_based"]),
        "total_skill_based": len(ROADMAP_CATEGORIES["skill_based"])
    }