import os
import torch
import numpy as np

def gen_explicit_from_file(text_file, tokenizer):
    # Generate prompts from a text file where the first line contains special flag tokens
    prompts = []
    token_indices = []
    
    with open(text_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                search_terms = line.strip().split(", ")

                flag_tokens = []
                for term in search_terms:
                    # Use add_special_tokens=False to get just the term tokens
                    check = tokenizer(" " + term, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
                    flag_tokens.append(check)

                continue
            
            prompt = line.strip()
            prompts.append(prompt)

            # Tokenize to find the index of the target token
            # Use add_special_tokens=True to match the actual tokenization used by the model
            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)['input_ids'][0]
            
            found = False
            for tok in flag_tokens:
                # Make sure the sequence matches
                for j in range(len(tokens) - len(tok) + 1):
                    if torch.equal(tokens[j:j+len(tok)], tok):
                        token_indices.append(j + len(tok) - 1)  # Index of last token in the term
                        found = True
                        break
                if found:
                    break
            
            if not found:
                print(f"WARNING: Could not find any flag token in prompt: {prompt[:50]}...")
                # Default to middle of the sentence if no flag token found
                token_indices.append(len(tokens) // 2)
                
    return prompts, token_indices


# Explicit velocity prompts for training linear probes
def gen_explicit_velocity(n_samples=500):
    prompts = []
    values = []
    
    objects = ["car", "train", "ball", "runner", "plane", "rocket", "vehicle", "object", "mass"]
    
    # Mix of simple and physics-style templates to match test distribution
    templates = [
        # Original simple templates (20%)
        "The {obj} is traveling at a speed of {v} m/s",
        "A {obj} moves at {v} m/s",
        
        # Physics-style templates with kinetic energy (40%)
        "A {m} kg {obj} has {ke} Joules of kinetic energy. The velocity is {v} m/s",
        "Given a {m} kg {obj} with {ke} Joules of kinetic energy, the velocity equals {v} m/s",
        "Mass: {m} kg, Kinetic Energy: {ke} Joules. The velocity is {v} m/s",
        "The {obj} weighs {m} kg and has {ke} Joules of kinetic energy, moving at {v} m/s",
        "A {m} kg {obj} with kinetic energy of {ke} J travels at a velocity of {v} m/s",
        "Consider a {m} kg {obj} possessing {ke} Joules of kinetic energy. Its velocity is {v} m/s",
        
        # Question-answer style (40%)
        "A {m} kg {obj} has {ke} Joules of kinetic energy. What is the velocity? {v} m/s",
        "Given mass {m} kg and kinetic energy {ke} J, the {obj} has velocity {v} m/s",
        "The {m} kg {obj} with {ke} Joules of kinetic energy. Velocity: {v} m/s",
        "Mass: {m} kg, KE: {ke} Joules. Calculate velocity: {v} m/s",
        "A {obj} weighing {m} kg with {ke} J kinetic energy moves at velocity {v} m/s",
    ]
    
    for _ in range(n_samples):
        obj = np.random.choice(objects)
        v = np.random.randint(2, 15)  # Match test range: 2-14 m/s
        template = np.random.choice(templates)
        
        # Generate physics variables if needed in template
        if "{m}" in template:
            m = np.random.randint(1, 10)  # mass in kg
            ke = int(0.5 * m * (v ** 2))  # kinetic energy
            prompt = template.format(obj=obj, v=v, m=m, ke=ke)
        else:
            prompt = template.format(obj=obj, v=v)
            
        prompts.append(prompt)
        values.append(float(v))
        
    return prompts, np.array(values)


# Explicit current prompts for training linear probes
def gen_explicit_current(n_samples=500):
    prompts = []
    values = []
    
    objects = ["circuit", "wire", "resistor", "device", "component", "conductor", "machine", "appliance"]
    
    # Mix of simple and physics-style templates to match test distribution
    templates = [
        # Original simple templates (20%)
        "The {obj} carries a current of {i} amperes",
        "A current of {i} A flows through the {obj}",
        
        # Physics-style templates with power and resistance (40%)
        "A {obj} has {r} ohms of resistance and dissipates {p} watts of power. The current is {i} A",
        "Given a {obj} with {r} ohms and {p} watts of power, the current equals {i} amperes",
        "Resistance: {r} ohms, Power: {p} watts. The current through the {obj} is {i} A",
        "The {obj} dissipates {p} watts across {r} ohms resistance, with current {i} A",
        "A {obj} with resistance {r} ohms and power output {p} W has a current of {i} amperes",
        "Consider a {obj} rated at {r} ohms dissipating {p} watts. Its current is {i} A",
        
        # Question-answer style (40%)
        "A {obj} has {r} ohms and dissipates {p} watts. What is the current? {i} amperes",
        "Given resistance {r} ohms and power {p} W, the {obj} carries current {i} A",
        "The {obj} with {r} ohms resistance and {p} watts power. Current: {i} A",
        "Resistance: {r} ohms, Power: {p} W. Calculate current: {i} amperes",
        "A {obj} with {r} ohms dissipating {p} watts has current {i} A flowing through it",
    ]
    
    for _ in range(n_samples):
        obj = np.random.choice(objects)
        i = np.random.randint(2, 15)  # Match test range: current between 2-15 A
        template = np.random.choice(templates)
        
        # Generate physics variables if needed in template
        if "{r}" in template:
            r = np.random.randint(1, 10)  # resistance in ohms
            p = i ** 2 * r  # power = I^2 * R
            prompt = template.format(obj=obj, i=i, r=r, p=p)
        else:
            prompt = template.format(obj=obj, i=i)
            
        prompts.append(prompt)
        values.append(float(i))
        
    return prompts, np.array(values)


# Moving mass with kinetic energy --> velocity (hidden) --> travel time
def gen_implicit_velocity(samples_per_prompt):
    prompts = []
    prompt_ids = []
    true_velocities = []  # Hidden variable that must be inferred

    objects = ["mass", "car", "train", "ball", "runner", "plane", "rocket", "vehicle"]
    prompt_formats = [
        "A {m} kg {obj} has {ke} Joules of kinetic energy. How long does it take to travel {d} m?",
        "Given a {m} kg {obj} with {ke} Joules of kinetic energy, calculate the duration required to cover a distance of {d} m.", 
        "The {obj} weighs {m} kg and possesses {ke} Joules of kinetic energy. What is the time needed to traverse {d} m?",
        "Mass: {m} kg, Kinetic Energy: {ke} Joules. Determine the time interval necessary for this {obj} to displace {d} m.",
        "Consider a {m} kg {obj} with {ke} Joules of kinetic energy. Find the number of seconds needed to move {d} m."
    ]
    
    for p_id, p_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            # We choose v and m first to ensure clean numbers, then calc KE
            v = np.random.randint(2, 100) # Keep numbers simple
            m = np.random.randint(1, 100)
            obj = np.random.choice(objects)
            ke = 0.5 * m * (v ** 2)
            d = np.random.randint(10, 100)  # Random distance for the travel time question
        
            # Prompt: "Mass is 2kg. Energy is {KE} Joules. Therefore the velocity is"
            # We want to see if the model has 'v' ready BEFORE it generates the number.
            prompt = p_format.format(m=m, obj=obj, ke=f"{ke:.3e}", d=d)
        
            prompts.append(prompt)
            prompt_ids.append(p_id)
            true_velocities.append(float(v))  # Store the true velocity value
        
    return prompts, prompt_ids, np.array(true_velocities)

# Resistor dissipating heat with power and resistance --> current (hidden) --> charge transferred after time t
def gen_implicit_current(samples_per_prompt):
    prompts = []
    prompt_ids = []
    true_currents = []  # Hidden variable that must be inferred

    objects = ["device", "computer", "appliance", "gadget", "machine", "resistor"]
    prompt_formats = [
        "A {obj} has {r} ohms of resistance and dissipates {p} watts of power. How much charge flows through it after {t} seconds?",
        "Given a {obj} with {r} ohms of resistance and {p} watts of power output, calculate the total charge in Coulombs that passes through over {t} seconds.", 
        "The {obj} dissipates {p} watts across a resistance of {r} ohms. Determine the magnitude of charge transferred during a {t} second interval.", 
        "Resistance: {r} ohms, Power: {p} watts. Find the net charge flow accumulated in this {obj} after {t} seconds.", 
        "Consider a {obj} rated at {r} ohms and {p} watts. How many Coulombs of charge travel through this component in {t} seconds?"
    ]
    
    for p_id, p_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            current = np.random.randint(2, 15)
            r = np.random.randint(1, 10)
            obj = np.random.choice(objects)
            power = current ** 2 * r  # P = I^2 * R
            t = np.random.randint(10, 100)  # time in seconds
        
            prompt = p_format.format(obj=obj, r=r, p=power, t=t)

            prompts.append(prompt)
            prompt_ids.append(p_id)
            true_currents.append(float(current))  # Store the true current value
        
    return prompts, prompt_ids, np.array(true_currents)

# Circle area --> radius or diameter (hidden) --> circumference
def gen_implicit_radius(samples_per_prompt):
    prompts = []
    prompt_ids = []
    true_radii = []  # Hidden variable that must be inferred
    
    shapes = ["circle", "circular disk", "round shape", "circular region", "disk", "circular area"]
    prompt_formats = [
        "A {shape} has an area of {area} square meters. What is its circumference in meters?",
        "Given a {shape} with a surface area of {area} square meters, calculate the perimeter.",
        "The {shape} covers {area} square meters of area. Determine the length around its edge in meters.",
        "Area: {area} square meters. Find the distance around the boundary of this {shape}.",
        "Consider a {shape} occupying {area} square meters. How many meters is the path around it?"
    ]
    
    for p_id, p_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            radius = np.random.randint(2, 20)  # radius in meters
            shape = np.random.choice(shapes)
            area = np.pi * radius ** 2  # A = πr²
            
            prompt = p_format.format(shape=shape, area=f"{area:.2f}")
            
            prompts.append(prompt)
            prompt_ids.append(p_id)
            true_radii.append(float(radius))
            
    return prompts, prompt_ids, np.array(true_radii)

# Cube volume --> side length (hidden) --> surface area
def gen_implicit_side_length(samples_per_prompt):
    prompts = []
    prompt_ids = []
    true_side_lengths = []  # Hidden variable that must be inferred
    
    objects = ["cube", "cubic box", "cubic container", "cubic block", "box", "cubic structure"]
    prompt_formats = [
        "A {obj} has a volume of {vol} cubic centimeters. What is its total surface area in square centimeters?",
        "Given a {obj} that contains {vol} cubic centimeters of space, calculate the surface area.",
        "The {obj} has an internal capacity of {vol} cubic centimeters. Determine its exterior surface area.",
        "Volume: {vol} cubic centimeters. Find the total area of all faces of this {obj} in square centimeters.",
        "Consider a {obj} with {vol} cubic centimeters of volume. How much surface area does it have?"
    ]
    
    for p_id, p_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            side = np.random.randint(2, 15)  # side length in cm
            obj = np.random.choice(objects)
            volume = side ** 3  # V = s³
            
            prompt = p_format.format(obj=obj, vol=volume)
            
            prompts.append(prompt)
            prompt_ids.append(p_id)
            true_side_lengths.append(float(side))
            
    return prompts, prompt_ids, np.array(true_side_lengths)

# Wave with speed and frequency --> wavelength (hidden) --> distance between n crests
def gen_implicit_wavelength(samples_per_prompt):
    prompts = []
    prompt_ids = []
    true_wavelengths = []  # Hidden variable that must be inferred
    
    wave_types = ["wave", "sound wave", "water wave", "acoustic wave", "wave pattern", "oscillation"]
    prompt_formats = [
        "A {wave} travels at {speed} m/s with a frequency of {freq} Hz. What is the distance between {n} consecutive crests?",
        "Given a {wave} moving at {speed} m/s and oscillating at {freq} Hz, calculate the span between {n} adjacent wave peaks.",
        "The {wave} has a speed of {speed} m/s and frequency of {freq} Hz. Determine the separation of {n} successive crests in meters.",
        "Speed: {speed} m/s, Frequency: {freq} Hz. Find the distance covered by {n} complete wavelengths of this {wave}.",
        "Consider a {wave} at {freq} Hz traveling at {speed} m/s. How many meters separate the first crest from the {n}th crest?"
    ]
    
    for p_id, p_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            wavelength = np.random.randint(2, 20)  # wavelength in meters
            freq = np.random.randint(2, 15)  # frequency in Hz
            wave_type = np.random.choice(wave_types)
            speed = wavelength * freq  # v = λf
            n_crests = np.random.randint(3, 8)  # number of wavelengths
            
            prompt = p_format.format(wave=wave_type, speed=speed, freq=freq, n=n_crests)
            
            prompts.append(prompt)
            prompt_ids.append(p_id)
            true_wavelengths.append(float(wavelength))
            
    return prompts, prompt_ids, np.array(true_wavelengths)

# Water through a cylindrical pipe with radius and speed --> Cross sectional area (hidden) --> volume after time t
def gen_implicit_cross_section(samples_per_prompt):
    prompts = []
    prompt_ids = []
    true_radii = []  # Hidden variable that must be inferred (radius, which determines cross-sectional area)
    
    conduits = ["pipe", "cylindrical pipe", "tube", "conduit", "circular pipe", "cylindrical tube"]
    prompt_formats = [
        "Water flows through a {conduit} with a radius of {r} cm at a speed of {v} cm/s. How much water (in cubic cm) passes through after {t} seconds?",
        "Given a {conduit} with radius {r} cm where water moves at {v} cm/s, calculate the volume of water in cubic centimeters that flows through for {t} seconds.",
        "The {conduit} has a {r} cm radius and water travels at {v} cm/s. Determine the total volume discharged in {t} seconds.",
        "Radius: {r} cm, Flow speed: {v} cm/s. Find the cubic centimeters of water transported through this {conduit} after {t} seconds.",
        "Consider a {conduit} of {r} cm radius carrying water at {v} cm/s. What volume in cubic cm flows through in {t} seconds?"
    ]
    
    for p_id, p_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            radius = np.random.randint(2, 15)  # radius in cm
            velocity = np.random.randint(5, 30)  # flow speed in cm/s
            conduit = np.random.choice(conduits)
            time = np.random.randint(10, 60)  # time in seconds
            
            prompt = p_format.format(conduit=conduit, r=radius, v=velocity, t=time)
            
            prompts.append(prompt)
            prompt_ids.append(p_id)
            true_radii.append(float(radius))
            
    return prompts, prompt_ids, np.array(true_radii)

# Spring with spring constant and force applied --> displacement (hidden) --> potential energy stored
def gen_implicit_displacement(samples_per_prompt):
    prompts = []
    prompt_ids = []
    true_displacements = []  # Hidden variable that must be inferred
    
    spring_types = ["spring", "elastic spring", "coil spring", "mechanical spring", "helical spring", "spring system"]
    prompt_formats = [
        "A {spring} with spring constant {k} N/m has {f} Newtons of force applied to it. What is the elastic potential energy stored in Joules?",
        "Given a {spring} with k = {k} N/m stretched by a force of {f} N, calculate the potential energy.",
        "The {spring} has stiffness {k} N/m and is under {f} N of force. Determine the stored elastic energy in Joules.",
        "Spring constant: {k} N/m, Applied force: {f} N. Find the potential energy stored in this {spring}.",
        "Consider a {spring} with spring constant {k} N/m pulled with {f} Newtons. How much elastic potential energy does it contain in Joules?"
    ]
    
    for p_id, p_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            displacement = np.random.randint(2, 15)  # displacement in meters (x)
            k = np.random.randint(5, 30)  # spring constant in N/m
            spring = np.random.choice(spring_types)
            force = k * displacement  # F = kx (Hooke's law)
            
            prompt = p_format.format(spring=spring, k=k, f=force)
            
            prompts.append(prompt)
            prompt_ids.append(p_id)
            true_displacements.append(float(displacement))
            
    return prompts, prompt_ids, np.array(true_displacements)

# Company with share price, number of shares and net income --> market cap (hidden) --> P/E ratio
def gen_implicit_market_cap(samples_per_prompt):
    prompts = []
    prompt_ids = []
    true_market_caps = []  # Hidden variable that must be inferred (in millions)
    
    company_types = ["company", "corporation", "firm", "business", "enterprise", "public company"]
    prompt_formats = [
        "A {company} has a share price of ${price} and {shares} million shares outstanding, with annual net income of ${income} million. What is its P/E ratio?",
        "Given a {company} trading at ${price} per share with {shares} million shares and ${income} million in net income, calculate the price-to-earnings ratio.",
        "The {company} has {shares} million shares at ${price} each and annual earnings of ${income} million. Determine the P/E ratio.",
        "Share price: ${price}, Shares outstanding: {shares} million, Net income: ${income} million. Find the P/E ratio for this {company}.",
        "Consider a {company} with {shares} million shares worth ${price} each that earns ${income} million annually. What is the price-to-earnings ratio?"
    ]
    
    for p_id, p_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            shares_millions = np.random.randint(50, 500)  # shares in millions
            price = np.random.randint(20, 200)  # price per share in dollars
            company = np.random.choice(company_types)
            market_cap = shares_millions * price  # market cap in millions
            # Generate income such that P/E ratio is reasonable (typically 10-30)
            pe_ratio = np.random.randint(10, 30)
            income = market_cap / pe_ratio  # earnings in millions
            
            prompt = p_format.format(company=company, price=price, shares=shares_millions, income=f"{income:.1f}")
            
            prompts.append(prompt)
            prompt_ids.append(p_id)
            true_market_caps.append(float(market_cap))
            
    return prompts, prompt_ids, np.array(true_market_caps)

