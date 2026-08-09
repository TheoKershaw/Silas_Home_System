import os
import sys
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# Made by Kershaw Technology

BLOCK_SIZE = 128       
N_EMBD = 128          
N_HEAD = 4             
N_LAYER = 4            
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_ITERS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 50

DEVICE = "cpu"  
torch.manual_seed(1337)

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def get_batch(data, batch_size=BATCH_SIZE):
    ix = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def train_model(text, save_path="model.pt"):
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    if len(train_data) <= BLOCK_SIZE or len(val_data) <= BLOCK_SIZE:
        print(
            "Your data.txt is quite small for the current BLOCK_SIZE. "
            "Add more text, or lower BLOCK_SIZE near the top of this file."
        )
        sys.exit(1)

    model = GPT(tokenizer.vocab_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters. Vocab size: {tokenizer.vocab_size}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {it:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(train_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "vocab_size": tokenizer.vocab_size,
        },
        save_path,
    )
    print(f"\nTraining complete. Model saved to {save_path}")
    return model, tokenizer


def load_model(save_path="model.pt"):
    checkpoint = torch.load(save_path, map_location=DEVICE, weights_only=False)
    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.stoi = checkpoint["stoi"]
    tokenizer.itos = checkpoint["itos"]
    tokenizer.vocab_size = checkpoint["vocab_size"]

    model = GPT(tokenizer.vocab_size).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer

def chat(model, tokenizer):
    print("\nChat with your model. Type 'quit' to exit.\n")
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in ("quit", "exit"):
            break

        prompt = f"User: {user_input}\nBot:"
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)

        out = model.generate(idx, max_new_tokens=200)
        generated = tokenizer.decode(out[0].tolist())

        reply = generated[len(prompt):]
        reply = reply.split("User:")[0].strip()
        print(f"Bot: {reply}\n")

PLACEHOLDER_DATA = """
Mechanical engineering is the study of physical machines and mechanisms that may involve force and movement. It is an engineering branch that combines engineering physics and mathematics principles with materials science, to design, analyze, manufacture, and maintain mechanical systems. It is one of the oldest and broadest of the engineering branches.
Mechanical engineering requires an understanding of core areas including mechanics, dynamics, thermodynamics, materials science, design, structural analysis, electronics, and electricity. In addition to these core principles, mechanical engineers use tools such as computer-aided design (CAD), computer-aided manufacturing (CAM), computer-aided engineering (CAE), and product lifecycle management to design and analyze manufacturing plants, industrial equipment and machinery, heating and cooling systems, transport systems, motor vehicles, aircraft, watercraft, robotics, medical devices, weapons, and others.
Mechanical engineering emerged as a field during the Industrial Revolution in Europe in the 18th century; however, its development can be traced back several thousand years around the world. In the 19th century, developments in physics led to the development of mechanical engineering science. The field has continually evolved to incorporate advancements; today mechanical engineers are pursuing developments in such areas as composites, mechatronics, and nanotechnology. It also overlaps with aerospace engineering, metallurgical engineering, civil engineering, structural engineering, electrical engineering, manufacturing engineering, chemical engineering, industrial engineering, and other engineering disciplines to varying amounts. Mechanical engineers may also work in the field of biomedical engineering, specifically with biomechanics, transport phenomena, biomechatronics, bionanotechnology, and modelling of biological systems.


== History ==

The application of mechanical engineering can be seen in the archives of various ancient and medieval societies. The six classic simple machines were known in the ancient Near East. The wedge and the inclined plane (ramp) were known since prehistoric times. Mesopotamian civilization is credited with the invention of the wheel by several, mainly old sources. However, some recent sources either suggest that it was invented independently in both Mesopotamia and Eastern Europe or credit prehistoric Eastern Europeans with the invention of the wheel The lever mechanism first appeared around 5,000 years ago in the Near East, where it was used in a simple balance scale, and to move large objects in ancient Egyptian technology. The lever was also used in the shadoof water-lifting device, the first crane machine, which appeared in Mesopotamia circa 3000 BC. The earliest evidence of pulleys date back to Mesopotamia in the early 2nd millennium BC.
The Saqiyah was developed in the Kingdom of Kush during the 4th century BC. It relied on animal power reducing the tow on the requirement of human energy. Reservoirs in the form of Hafirs were developed in Kush to store water and boost irrigation. Bloomeries and blast furnaces were developed during the seventh century BC in Meroe. Kushite sundials applied mathematics in the form of advanced trigonometry.
The earliest practical water-powered machines, the water wheel and watermill, first appeared in the Persian Empire, in what are now Iraq and Iran, by the early 4th century BC. In ancient Greece, the works of Archimedes (287–212 BC) influenced mechanics in the Western tradition. The geared Antikythera mechanisms was an Analog computer invented around the 2nd century BC.
In Roman Egypt, Heron of Alexandria (c. 10–70 AD) created the first steam-powered device (Aeolipile). In China, Zhang Heng (78–139 AD) improved a water clock and invented a seismometer, and Ma Jun (200–265 AD) invented a chariot with differential gears. The medieval Chinese horologist and engineer Su Song (1020–1101 AD) incorporated an escapement mechanism into his astronomical clock tower two centuries before escapement devices were found in medieval European clocks. He also invented the world's first known endless power-transmitting chain drive.
The cotton gin was invented in India by the 6th century AD, and the spinning wheel was invented in the Islamic world by the early 11th century, Dual-roller gins appeared in India and China between the 12th and 14th centuries. The worm gear roller gin appeared in the Indian subcontinent during the early Delhi Sultanate era of the 13th to 14th centuries.
During the Islamic Golden Age (7th to 15th century), Muslim inventors made remarkable contributions in the field of mechanical technology. Al-Jazari, who was one of them, wrote his famous Book of Knowledge of Ingenious Mechanical Devices in 1206 and presented many mechanical designs.
In the 17th century, important breakthroughs in the foundations of mechanical engineering occurred in England and the Continent. The Dutch mathematician and physicist Christiaan Huygens invented the pendulum clock in 1657, which was the first reliable timekeeper for almost 300 years, and published a work dedicated to clock designs and the theory behind them. In England, Isaac Newton formulated his laws of motion and developed calculus, which would become the mathematical basis of physics. Newton was reluctant to publish his works for years, but he was finally persuaded to do so by his colleagues, such as Edmond Halley. Gottfried Wilhelm Leibniz, who earlier designed a mechanical calculator, is also credited with developing the calculus during the same time period.
During the early 19th century Industrial Revolution, machine tools were developed in England, Germany, and Scotland. This allowed mechanical engineering to develop as a separate field within engineering. They brought with them manufacturing machines and the engines to power them. The first British professional society of mechanical engineers was formed in 1847 Institution of Mechanical Engineers, thirty years after the civil engineers formed the first such professional society Institution of Civil Engineers. On the European continent, Johann von Zimmermann (1820–1901) founded the first factory for grinding machines in Chemnitz, Germany in 1848.
In the United States, the American Society of Mechanical Engineers (ASME) was formed in 1880, becoming the third such professional engineering society, after the American Society of Civil Engineers (1852) and the American Institute of Mining Engineers (1871). The first schools in the US to offer an engineering education were the United States Military Academy in 1817, an institution now known as Norwich University in 1819, and Rensselaer Polytechnic Institute in 1825. Education in mechanical engineering has historically been based on a strong foundation in mathematics and science.


== Education ==

Degrees in mechanical engineering are offered at various universities worldwide. Mechanical engineering programs typically take four to five years of study depending on the place and university and result in a Bachelor of Engineering (B.Eng. or B.E.), Bachelor of Science (B.Sc. or B.S.), Bachelor of Science Engineering (B.Sc.Eng.), Bachelor of Technology (B.Tech.), Bachelor of Mechanical Engineering (B.M.E.), or Bachelor of Applied Science (B.A.Sc.) degree, in or with emphasis in mechanical engineering. In Spain, Portugal and most of South America, where neither B.S. nor B.Tech. programs have been adopted, the formal name for the degree is "Mechanical Engineer", and the course work is based on five or six years of training. In Italy the course work is based on five years of education, and training, but in order to qualify as an Engineer one has to pass a state exam at the end of the course. In Greece, the coursework is based on a five-year curriculum.
In the United States, most undergraduate mechanical engineering programs are accredited by the Accreditation Board for Engineering and Technology (ABET) to ensure similar course requirements and standards among universities. The ABET web site lists 302 accredited mechanical engineering programs as of 11 March 2014. Mechanical engineering programs in Canada are accredited by the Canadian Engineering Accreditation Board (CEAB), and most other countries offering engineering degrees have similar accreditation societies.
In Australia, mechanical engineering degrees are awarded as Bachelor of Engineering (Mechanical) or similar nomenclature, although there are an increasing number of specialisations. The degree takes four years of full-time study to achieve. To ensure quality in engineering degrees, Engineers Australia accredits engineering degrees awarded by Australian universities in accordance with the global Washington Accord. Before the degree can be awarded, the student must complete at least 3 months of on the job work experience in an engineering firm. Similar systems are also present in South Africa and are overseen by the Engineering Council of South Africa (ECSA).
In India, to become an engineer, one needs to have an engineering degree like a B.Tech. or B.E., have a diploma in engineering, or by completing a course in an engineering trade like fitter from the Industrial Training Institute (ITIs) to receive a "ITI Trade Certificate" and also pass the All India Trade Test (AITT) with an engineering trade conducted by the National Council of Vocational Training (NCVT) by which one is awarded a "National Trade Certificate". A similar system is used in Nepal.
Some mechanical engineers go on to pursue a postgraduate degree such as a Master of Engineering, Master of Technology, Master of Science, Master of Engineering Management (M.Eng.Mgt. or M.E.M.), a Doctor of Philosophy in engineering (Eng.D. or Ph.D.) or an engineer's degree. The master's and engineer's degrees may or may not include research. The Doctor of Philosophy includes a significant research component and is often viewed as the entry point to academia. The Engineer's degree exists at a few institutions at an intermediate level between the master's degree and the doctorate.


=== Coursework ===
Standards set by each country's accreditation society are intended to provide uniformity in fundamental subject material, promote competence among graduating engineers, and to maintain confidence in the engineering profession as a whole. Engineering programs in the U.S., for example, are required by ABET to show that their students can "work professionally in both thermal and mechanical systems areas." The specific courses required to graduate, however, may differ from program to program. Universities and institutes of technology will often combine multiple subjects into a single class or split a subject into multiple classes, depending on the faculty available and the university's major area(s) of research.
The fundamental subjects required for mechanical engineering usually include:

Mathematics (in particular, calculus, differential equations, and linear algebra)
Basic physical sciences (including physics and chemistry)
Statics and dynamics
Strength of materials and solid mechanics
Materials engineering, composites
Thermodynamics, heat transfer, energy conversion, and HVAC
Fuels, combustion, internal combustion engine
Fluid mechanics (including fluid statics and fluid dynamics)
Mechanism and Machine design (including kinematics and dynamics)
Instrumentation and measurement
Manufacturing engineering, technology, or processes
Vibration, control theory and control engineering
Hydraulics and Pneumatics
Mechatronics and robotics
Engineering design and product design
Drafting, computer-aided design (CAD) and computer-aided manufacturing (CAM)
Mechanical engineers are also expected to understand and be able to apply basic concepts from chemistry, physics, tribology, chemical engineering, civil engineering, and electrical engineering. All mechanical engineering programs include multiple semesters of mathematical classes including calculus, and advanced mathematical concepts including differential equations, partial differential equations, linear algebra, differential geometry, and statistics, among others.
In addition to the core mechanical engineering curriculum, many mechanical engineering programs offer more specialized programs and classes, such as control systems, robotics, transport and logistics, cryogenics, fuel technology, automotive engineering, biomechanics, vibration, optics and others, if a separate department does not exist for these subjects.
Most mechanical engineering programs also require varying amounts of research or community projects to gain practical problem-solving experience. In the US it is common for mechanical engineering students to complete one or more internships while studying, though this is not typically mandated by the university. Cooperative education is another option. Research puts demand on study components that feed student's creativity and innovation.


== Job duties ==
Mechanical engineers research, design, develop, build, and test mechanical and thermal devices, including tools, engines, and machines.
Mechanical engineers typically do the following:

Analyze problems to see how mechanical and thermal devices might help solve the problem.
Design or redesign mechanical and thermal devices using analysis and computer-aided design.
Develop and test prototypes of devices they design.
Analyze the test results and change the design as needed.
Oversee the manufacturing process for the device.
Manage a team of professionals in specialized fields like mechanical drafting and designing, prototyping, 3D printing or/and CNC Machines specialists.
Mechanical engineers design and oversee the manufacturing of many products ranging from medical devices to new batteries. They also design power-producing machines such as electric generators, internal combustion engines, and steam and gas turbines as well as power-using machines, such as refrigeration and air-conditioning systems.
Like other engineers, mechanical engineers use computers to help create and analyze designs, run simulations and test how a machine is likely to work.


=== License and regulation ===
Engineers may seek license by a state, provincial, or national government. The purpose of this process is to ensure that engineers possess the necessary technical knowledge, real-world experience, and knowledge of the local legal system to practice engineering at a professional level. Once certified, the engineer is given the title of Professional Engineer (in the US, Canada, Japan, South Korea, Bangladesh and South Africa). Chartered Engineer is used in the United Kingdom, Ireland, India and Zimbabwe, along with Chartered Professional Engineer (in Australia and New Zealand) or European Engineer (much of the European Union).
In the U.S., to become a licensed Professional Engineer (PE), an engineer must pass the comprehensive FE (Fundamentals of Engineering) exam, work a minimum of 4 years as an Engineering Intern (EI) or Engineer-in-Training (EIT), and pass the "Principles and Practice" or PE (Practicing Engineer or Professional Engineer) exams. The requirements and steps of this process are set forth by the National Council of Examiners for Engineering and Surveying (NCEES), composed of engineering and land surveying licensing boards representing all U.S. states and territories.
In Australia (Queensland and Victoria) an engineer must be registered as a Professional Engineer within the State in which they practice, for example Registered Professional Engineer of Queensland or Victoria, RPEQ or RPEV. respectively.
In the UK, current graduates require a BEng plus an appropriate master's degree or an integrated MEng degree, a minimum of 4 years post graduate on the job competency development and a peer-reviewed project report to become a Chartered Mechanical Engineer (CEng, MIMechE) through the Institution of Mechanical Engineers. CEng MIMechE can also be obtained via an examination route administered by the City and Guilds of London Institute.
In most developed countries, certain engineering tasks, such as the design of bridges, electric power plants, and chemical plants, must be approved by a professional engineer or a chartered engineer. "Only a licensed engineer, for instance, may prepare, sign, seal and submit engineering plans and drawings to a public authority for approval, or to seal engineering work for public and private clients." This requirement can be written into state and provincial legislation, such as in the Canadian provinces, for example the Ontario or Quebec's Engineer Act.
In other countries, such as the UK, no such legislation exists; however, practically all certifying bodies maintain a code of ethics independent of legislation, that they expect all members to abide by or risk expulsion.


=== Salaries and workforce statistics ===
The total number of engineers employed in the U.S. in 2015 was roughly 1.6 million. Of these, 278,340 were mechanical engineers (17.28%), the largest discipline by size. In 2012, the median annual income of mechanical engineers in the U.S. workforce was $80,580. The median income was highest when working for the government ($92,030), and lowest in education ($57,090). In 2014, the total number of mechanical engineering jobs was projected to grow 5% over the next decade. As of 2009, the average starting salary was $58,800 with a bachelor's degree.


== Subdisciplines ==
The field of mechanical engineering can be thought of as a collection of many mechanical engineering science disciplines. Several of these subdisciplines which are typically taught at the undergraduate level are listed below, with a brief explanation and the most common application of each. Some of these subdisciplines are unique to mechanical engineering, while others are a combination of mechanical engineering and one or more other disciplines. Most work that a mechanical engineer does uses skills and techniques from several of these subdisciplines, as well as specialized subdisciplines. Specialized subdisciplines, as used in this article, are more likely to be the subject of graduate studies or on-the-job training than undergraduate research. Several specialized subdisciplines are discussed in this section.


=== Mechanics ===

Mechanics is, in the most general sense, the study of forces and their effect upon matter. Typically, engineering mechanics is used to analyze and predict the acceleration and deformation (both elastic and plastic) of objects under known forces (also called loads) or stresses. Subdisciplines of mechanics include

Statics, the study of non-moving bodies under known loads, how forces affect static bodies
Dynamics, the study of how forces affect moving bodies. Dynamics includes kinematics (about movement, velocity, and acceleration) and kinetics (about forces and resulting accelerations).
Mechanics of materials, the study of how different materials deform under various types of stress
Fluid mechanics, the study of how fluids react to forces
Kinematics, the study of the motion of bodies (objects) and systems (groups of objects), while ignoring the forces that cause the motion. Kinematics is often used in the design and analysis of mechanisms.
Continuum mechanics, a method of applying mechanics that assumes that objects are continuous (rather than discrete)
Mechanical engineers typically use mechanics in the design or analysis phases of engineering. If the engineering project were the design of a vehicle, statics might be employed to design the frame of the vehicle, in order to evaluate where the stresses will be most intense. Dynamics might be used when designing the car's engine, to evaluate the forces in the pistons and cams as the engine cycles. Mechanics of materials might be used to choose appropriate materials for the frame and engine. Fluid mechanics might be used to design a ventilation system for the vehicle (see HVAC), or to design the intake system for the engine.


=== Mechatronics and robotics ===

Mechatronics is a combination of mechanics and electronics. It is an interdisciplinary branch of mechanical engineering, electrical engineering and software engineering that is concerned with integrating electrical and mechanical engineering to create hybrid automation systems. In this way, machines can be automated through the use of electric motors, servo-mechanisms, and other electrical systems in conjunction with special software. A common example of a mechatronics system is a CD-ROM drive. Mechanical systems open and close the drive, spin the CD and move the laser, while an optical system reads the data on the CD and converts it to bits. Integrated software controls the process and communicates the contents of the CD to the computer.
Robotics is the application of mechatronics to create robots, which are often used in industry to perform tasks that are dangerous, unpleasant, or repetitive. These robots may be of any shape and size, but all are preprogrammed and interact physically with the world. To create a robot, an engineer typically employs kinematics (to determine the robot's range of motion) and mechanics (to determine the stresses within the robot).
Robots are used extensively in industrial automation engineering. They allow businesses to save money on labor, perform tasks that are either too dangerous or too precise for humans to perform them economically, and to ensure better quality. Many companies employ assembly lines of robots, especially in Automotive Industries and some factories are so robotized that they can run by themselves. Outside the factory, robots have been employed in bomb disposal, space exploration, and many other fields. Robots are also sold for various residential applications, from recreation to domestic applications.


=== Structural analysis ===

Structural analysis is the branch of mechanical engineering (and also civil engineering) devoted to examining why and how objects fail and to fix the objects and their performance. Structural failures occur in two general modes: static failure, and fatigue failure. Static structural failure occurs when, upon being loaded (having a force applied) the object being analyzed either breaks or is deformed plastically, depending on the criterion for failure. Fatigue failure occurs when an object fails after a number of repeated loading and unloading cycles. Fatigue failure occurs because of imperfections in the object: a microscopic crack on the surface of the object, for instance, will grow slightly with each cycle (propagation) until the crack is large enough to cause ultimate failure.
Failure is not simply defined as when a part breaks, however; it is defined as when a part does not operate as intended. Some systems, such as the perforated top sections of some plastic bags, are designed to break. If these systems do not break, failure analysis might be employed to determine the cause.
Structural analysis is often used by mechanical engineers after a failure has occurred, or when designing to prevent failure. Engineers often use online documents and books such as those published by ASM to aid them in determining the type of failure and possible causes.
Once theory is applied to a mechanical design, physical testing is often performed to verify calculated results. Structural analysis may be used in an office when designing parts, in the field to analyze failed parts, or in laboratories where parts might undergo controlled failure tests.


=== Thermodynamics and thermal engineering ===

Thermodynamics is a physical science concerned with heat, work, energy, and the physical properties of matter. It is used in several branches of engineering, including mechanical and chemical engineering. Engineering thermodynamics focuses on systems that change energy from one form to another. 
More broadly, thermal engineering combines thermodynamics and the science of transport phenomena to study heat transfer, combustion, and compressible fluid flow in engineered systems. Mechanical engineers use thermal engineering to design equipment such as engines, power plants, heat exchangers, heat sinks, radiators, refrigerators, insulation, and heating, ventilation, and air-conditioning (HVAC) systems.
As an example, automotive engines convert the chemical energy of fuel (enthalpy) into heat and then into the mechanical work that turns a car's wheels. A mechanical engineer will aim for an engine design that maximizes the fraction of the initial chemical energy that is converted to mechanical work, rather than being lost as heat or friction. An ideal design would approach Carnot efficiency. (In fact, Sadi Carnot was a French military engineer.)


=== Design and drafting ===

Drafting or technical drawing is the means by which mechanical engineers design products and create instructions for manufacturing parts. A technical drawing can be a computer model or hand-drawn schematic showing all the dimensions necessary to manufacture a part, as well as assembly notes, a list of required materials, and other pertinent information. A U.S. mechanical engineer or skilled worker who creates technical drawings may be referred to as a drafter or draftsman. Drafting has historically been a two-dimensional process, but computer-aided design (CAD) programs now allow the designer to create in three dimensions.
Instructions for manufacturing a part must be fed to the necessary machinery, either manually, through programmed instructions, or through the use of a computer-aided manufacturing (CAM) or combined CAD/CAM program. Optionally, an engineer may also manually manufacture a part using the technical drawings. However, with the advent of computer numerically controlled (CNC) manufacturing, parts can now be fabricated without the need for constant technician input. Manually manufactured parts generally consist of spray coatings, surface finishes, and other processes that cannot economically or practically be done by a machine.
Drafting is used in nearly every subdiscipline of mechanical engineering, and by many other branches of engineering and architecture. Three-dimensional models created using CAD software are also commonly used in finite element analysis (FEA) and computational fluid dynamics (CFD).


== Modern tools ==


=== Computer aided software suites ===
Many mechanical engineering companies, especially those in industrialized nations, have incorporated computer-aided engineering (CAE) programs into their existing design and analysis processes, including 2D and 3D solid modeling computer-aided design (CAD). This method has many benefits, including easier and more exhaustive visualization of products, the ability to create virtual assemblies of parts, and the ease of use in designing mating interfaces and tolerances.
Other CAE programs commonly used by mechanical engineers include product lifecycle management (PLM) tools and analysis tools used to perform complex simulations. Analysis tools may be used to predict product response to expected loads, including fatigue life and manufacturability. These tools include finite element analysis (FEA), computational fluid dynamics (CFD), and computer-aided manufacturing (CAM).

Using CAE programs, a mechanical design team can quickly and cheaply iterate the design process to develop a product that better meets cost, performance, and other constraints. No physical prototype need be created until the design nears completion, allowing hundreds or thousands of designs to be evaluated, instead of a relative few. In addition, CAE analysis programs can model complicated physical phenomena which cannot be solved by hand, such as viscoelasticity, complex contact between mating parts, or non-Newtonian flows. 
As mechanical engineering begins to merge with other disciplines, as seen in mechatronics, multidisciplinary design optimization (MDO) is being used with other CAE programs to automate and improve the iterative design process. MDO tools wrap around existing CAE processes, allowing product evaluation to continue even after the analyst goes home for the day. They also use sophisticated optimization algorithms to more intelligently explore possible designs, often finding better, innovative solutions to difficult multidisciplinary design problems.


=== On‑demand platforms for external FEA expertise ===
Engineering teams can access external finite‑element analysis (FEA) expertise through on‑demand platforms. By submitting structured project inputs—such as CAD or FEA input files, load cases, boundary conditions and desired deliverables - users receive instant quotes and connect with pre‑qualified simulation engineers. Providers manage the entire workflow remotely, often running simulations on cloud and GPU‑based infrastructure to accelerate turnaround, which can be significantly faster than in‑house CPU‑based systems. This model offers scalability, access to niche simulation domains like CFD or multiphysics, and avoids the delays, licensing burden and overhead of traditional procurement cycles. It is particularly valuable for smaller teams operating under tight deadlines or limited in‑house capacity.


== Areas of research ==
Mechanical engineers are constantly pushing the boundaries of what is physically possible in order to produce safer, cheaper, and more efficient machines and mechanical systems. Some technologies at the cutting edge of mechanical engineering are listed below (see also exploratory engineering).


=== Micro electro-mechanical systems (MEMS) ===

Micron-scale mechanical components such as springs, gears, fluidic and heat transfer devices are fabricated from a variety of substrate materials such as silicon, glass and polymers like SU8. Examples of MEMS components are the accelerometers that are used as car airbag sensors, modern cell phones, gyroscopes for precise positioning and microfluidic devices used in biomedical applications.


=== Friction stir welding (FSW) ===

Friction stir welding, a new type of welding, was discovered in 1991 by The Welding Institute (TWI). The innovative steady state (non-fusion) welding technique joins materials previously un-weldable, including several aluminum alloys. It plays an important role in the future construction of airplanes, potentially replacing rivets. Current uses of this technology to date include welding the seams of the aluminum main Space Shuttle external tank, Orion Crew Vehicle, Boeing Delta II and Delta IV Expendable Launch Vehicles and the SpaceX Falcon 1 rocket, armor plating for amphibious assault ships, and welding the wings and fuselage panels of the new Eclipse 500 aircraft from Eclipse Aviation among an increasingly growing pool of uses.


=== Composites ===

Composites or composite materials are a combination of materials which provide different physical characteristics than either material separately. Composite material research within mechanical engineering typically focuses on designing (and, subsequently, finding applications for) stronger or more rigid materials while attempting to reduce weight, susceptibility to corrosion, and other undesirable factors. Carbon fiber reinforced composites, for instance, have been used in such diverse applications as spacecraft and fishing rods.


=== Mechatronics ===
Mechatronics is the synergistic combination of mechanical engineering, electronic engineering, and software engineering. The discipline of mechatronics began as a way to combine mechanical principles with electrical engineering. Mechatronic concepts are used in the majority of electro-mechanical systems. Typical electro-mechanical sensors used in mechatronics are strain gauges, thermocouples, and pressure transducers.


=== Nanotechnology ===

At the smallest scales, mechanical engineering becomes nanotechnology. One speculative goal is to create a molecular assembler to build molecules and materials via mechanosynthesis; however, that goal remains within exploratory engineering. Areas of current mechanical engineering research in nanotechnology include nanofilters, nanofilms, and nanostructures, among others.


=== Finite element analysis ===

Finite Element Analysis is a computational tool used to estimate stress, strain, and deflection of solid bodies. It uses a mesh setup with user-defined sizes to measure physical quantities at a node. The more nodes there are, the higher the precision. This field is not new, as the basis of Finite Element Analysis (FEA) or Finite Element Method (FEM) dates back to 1941. But the evolution of computers has made FEA/FEM a viable option for analysis of structural problems. Many commercial software applications such as NASTRAN, ANSYS, and ABAQUS are widely used in industry for research and the design of components. Some 3D modeling and CAD software packages have added FEA modules. In the recent times, cloud simulation platforms like SimScale are becoming more common.
Other techniques such as finite difference method (FDM) and finite-volume method (FVM) are employed to solve problems relating heat and mass transfer, fluid flows, fluid surface interaction, etc.


=== Biomechanics ===

Biomechanics is the application of mechanical principles to biological systems, such as humans, animals, plants, organs, and cells. Biomechanics also aids in creating prosthetic limbs and artificial organs for humans. Biomechanics is closely related to engineering, because it often uses traditional engineering sciences to analyze biological systems. Some simple applications of Newtonian mechanics and/or materials sciences can supply correct approximations to the mechanics of many biological systems.
In the past decade, reverse engineering of materials found in nature such as bone matter has gained funding in academia. The structure of bone matter is optimized for its purpose of bearing a large amount of compressive stress per unit weight. The goal is to replace crude steel with bio-material for structural design.
Over the past decade the finite element method (FEM) has also entered the Biomedical sector highlighting further engineering aspects of Biomechanics. FEM has since then established itself as an alternative to in vivo surgical assessment and gained the wide acceptance of academia. The main advantage of Computational Biomechanics lies in its ability to determine the endo-anatomical response of an anatomy, without being subject to ethical restrictions. This has led FE modelling to the point of becoming ubiquitous in several fields of Biomechanics while several projects have even adopted an open source philosophy (e.g. BioSpine).


=== Computational fluid dynamics ===

Computational fluid dynamics, usually abbreviated as CFD, is a branch of fluid mechanics that uses numerical methods and algorithms to solve and analyze problems that involve fluid flows. Computers are used to perform the calculations required to simulate the interaction of liquids and gases with surfaces defined by boundary conditions. With high-speed supercomputers, better solutions can be achieved. Ongoing research yields software that improves the accuracy and speed of complex simulation scenarios such as turbulent flows. Initial validation of such software is performed using a wind tunnel with the final validation coming in full-scale testing, e.g. flight tests.


=== Acoustical engineering ===

Acoustical engineering is one of many other sub-disciplines of mechanical engineering and is the application of acoustics. Acoustical engineering is the study of sound and vibration. Acoustical engineers work to reduce noise pollution in mechanical devices and in buildings by soundproofing or removing sources of unwanted noise. The study of acoustics can range from designing a more efficient hearing aid, microphone, headphone, or recording studio to enhancing the sound quality of an orchestra hall. Acoustical engineering also deals with the vibration of different mechanical systems.


== Related fields ==
Manufacturing engineering, aerospace engineering, automotive engineering and marine engineering are grouped with mechanical engineering at times. A bachelor's degree in these areas will typically have a difference of a few specialized classes.


== See also ==

Automobile engineering
Index of mechanical engineering articles
Lists

Associations

Wikibooks


== References ==


== Further reading ==

Burstall, Aubrey F. (1965). A History of Mechanical Engineering. The MIT Press. ISBN 978-0-262-52001-0.
Marks' Standard Handbook for Mechanical Engineers (11 ed.). McGraw-Hill. 2007. ISBN 978-0-07-142867-5.
Oberg, Erik; Franklin D. Jones; Holbrook L. Horton; Henry H. Ryffel; Christopher McCauley (2016). Machinery's Handbook (30th ed.). New York: Industrial Press Inc. ISBN 978-0-8311-3091-6.


== External links ==

Mechanical engineering at MTU.edu

Thermodynamics is a branch of physics that deals with heat, work, and temperature, and their relation to energy, entropy, and the physical properties of matter and radiation. The behavior of these quantities is governed by the four laws of thermodynamics, which convey a quantitative description using measurable macroscopic physical quantities but may be explained in terms of microscopic constituents by statistical mechanics. Thermodynamics applies to various topics in science and engineering, especially physical chemistry, biochemistry, chemical engineering, and mechanical engineering, as well as other complex fields such as meteorology.
Historically, thermodynamics developed out of a desire to increase the efficiency of early steam engines, particularly through the work of French physicist Sadi Carnot (1824). Scots-Irish physicist Lord Kelvin was the first to formulate a concise definition of thermodynamics in 1854 which stated, "Thermo-dynamics is the subject of the relation of heat to forces acting between contiguous parts of bodies, and the relation of heat to electrical agency."  German physicist and mathematician Rudolf Clausius restated Carnot's principle known as the Carnot cycle and gave the theory of heat a more accurate and sounder basis. His most important paper, "On the Moving Force of Heat", published in 1850, first stated the second law of thermodynamics. In 1865 he named the concept of entropy. In 1870 he introduced the virial theorem, which applied to heat.
The initial application of thermodynamics to mechanical heat engines was quickly extended to the study of chemical compounds and chemical reactions. Chemical thermodynamics studies the nature of the role of entropy in the process of chemical reactions and has provided the bulk of expansion and knowledge of the field. Other formulations of thermodynamics emerged. Statistical thermodynamics, or statistical mechanics, concerns itself with statistical predictions of the collective motion of particles from their microscopic behavior. In 1909, Constantin Carathéodory presented a purely mathematical approach in an axiomatic formulation, a description often referred to as geometrical thermodynamics.


== Introduction ==
A description of any thermodynamic system employs the four laws of thermodynamics that form an axiomatic basis. The first law specifies that energy can be transferred between physical systems as heat, as work, and with the transfer of matter. The second law defines the existence of a quantity called entropy, which describes the direction, thermodynamically, that a system can evolve and quantifies the state of order of a system and which can be used to quantify the useful work that can be extracted from the system.
In thermodynamics, interactions between large ensembles of objects are studied and categorized. Central to this are the concepts of the thermodynamic system and its surroundings. A system is composed of particles, whose average motions define its properties, and those properties are in turn related to one another through equations of state. Properties can be combined to express internal energy and thermodynamic potentials, which are useful for determining conditions for equilibrium and spontaneous processes.
With these tools, thermodynamics can be used to describe how systems respond to changes in their environment. This can be applied to a wide variety of topics in science and engineering, such as engines, phase transitions, chemical reactions, transport phenomena, and even black holes. The results of thermodynamics are essential for other fields of physics and for chemistry, chemical engineering, corrosion engineering, aerospace engineering, mechanical engineering, cell biology, biomedical engineering, materials science, and economics, to name a few.
This article is focused mainly on classical thermodynamics which primarily studies systems in thermodynamic equilibrium. Non-equilibrium thermodynamics is often treated as an extension of the classical treatment, but statistical mechanics has brought many advances to that field.


== History ==
The history of thermodynamics as a scientific discipline generally begins with Otto von Guericke who, in 1650, built and designed the world's first vacuum pump and demonstrated a vacuum using his Magdeburg hemispheres. Guericke was driven to make a vacuum in order to disprove Aristotle's long-held supposition that 'nature abhors a vacuum'. Shortly after Guericke, the Anglo-Irish physicist and chemist Robert Boyle had learned of Guericke's designs and, in 1656, in coordination with English scientist Robert Hooke, built an air pump. Using this pump, Boyle and Hooke noticed a correlation between pressure, temperature, and volume. In time, Boyle's Law was formulated, which states that pressure and volume are inversely proportional. Then, in 1679, based on these concepts, an associate of Boyle's named Denis Papin built a steam digester, which was a closed vessel with a tightly fitting lid that confined steam until a high pressure was generated.
Later designs implemented a steam release valve that kept the machine from exploding. By watching the valve rhythmically move up and down, Papin conceived of the idea of a piston and a cylinder engine. He did not, however, follow through with his design. Nevertheless, in 1697, based on Papin's designs, engineer Thomas Savery built the first engine, followed by Thomas Newcomen in 1712. Although these early engines were crude and inefficient, they attracted the attention of the leading scientists of the time.
The fundamental concepts of heat capacity and latent heat, which were necessary for the development of thermodynamics, were developed by Professor Joseph Black at the University of Glasgow, where James Watt was employed as an instrument maker. Black and Watt performed experiments together, but it was Watt who conceived the idea of the external condenser which resulted in a large increase in steam engine efficiency. Drawing on all the previous work led Sadi Carnot, the "father of thermodynamics", to publish Reflections on the Motive Power of Fire (1824), a discourse on heat, power, energy and engine efficiency. The book outlined the basic energetic relations between the Carnot engine, the Carnot cycle, and motive power. It marked the start of thermodynamics as a modern science.
The first thermodynamic textbook was written in 1859 by William Rankine, originally trained as a physicist and a civil and mechanical engineering professor at the University of Glasgow. The first and second laws of thermodynamics emerged simultaneously in the 1850s, primarily out of the works of William Rankine, Rudolf Clausius, and William Thomson (Lord Kelvin).
The foundations of statistical thermodynamics were set out by physicists such as James Clerk Maxwell, Ludwig Boltzmann, Max Planck, Rudolf Clausius and J. Willard Gibbs.
Clausius, who first stated the basic ideas of the second law in his paper "On the Moving Force of Heat", published in 1850, and is called "one of the founding fathers of thermodynamics", introduced the concept of entropy in 1865.
During the years 1873–76 the American mathematical physicist Josiah Willard Gibbs published a series of three papers, the most famous being On the Equilibrium of Heterogeneous Substances, in which he showed how thermodynamic processes, including chemical reactions, could be graphically analyzed, by studying the energy, entropy, volume, temperature and pressure of the thermodynamic system in such a manner, one can determine if a process would occur spontaneously. Also Pierre Duhem in the 19th century wrote about chemical thermodynamics. During the early 20th century, chemists such as Gilbert N. Lewis, Merle Randall, and E. A. Guggenheim applied the mathematical methods of Gibbs to the analysis of chemical processes.


== Etymology ==
Thermodynamics has an intricate etymology.
By a surface-level analysis, the word consists of two parts that can be traced back to Ancient Greek. Firstly, thermo- ("of heat"; used in words such as thermometer) can be traced back to the root θέρμη therme, meaning "heat". Secondly, the word dynamics ("science of force [or power]") can be traced back to the root δύναμις dynamis, meaning "power".
In 1849, the adjective thermo-dynamic is used by William Thomson.
In 1854, the noun thermo-dynamics is used by Thomson and William Rankine to represent the science of generalized heat engines.
Pierre Perrot claims that the term thermodynamics was coined by James Joule in 1858 to designate the science of relations between heat and power, however, Joule never used that term, but used instead the term perfect thermo-dynamic engine in reference to Thomson's 1849 phraseology.


== Branches of thermodynamics ==
The study of thermodynamic systems has developed into several related branches, each using a different fundamental model as a theoretical or experimental basis, or applying the principles to varying types of systems.


=== Classical thermodynamics ===
Classical thermodynamics is the description of the states of thermodynamic systems at near-equilibrium, that uses macroscopic, measurable properties. It is used to model exchanges of energy, work and heat based on the laws of thermodynamics. The qualifier classical reflects the fact that it represents the first level of understanding of the subject as it developed in the 19th century and describes the changes of a system in terms of macroscopic empirical (large scale, and measurable) parameters. A microscopic interpretation of these concepts was later provided by the development of statistical mechanics.


=== Statistical mechanics ===
Statistical mechanics, also known as statistical thermodynamics, emerged with the development of atomic and molecular theories in the late 19th century and early 20th century, and supplemented classical thermodynamics with an interpretation of the microscopic interactions between individual particles or quantum-mechanical states. This field relates the microscopic properties of individual atoms and molecules to the macroscopic, bulk properties of materials that can be observed on the human scale, thereby explaining classical thermodynamics as a natural result of statistics, classical mechanics, and quantum theory at the microscopic level.


=== Chemical thermodynamics ===
Chemical thermodynamics is the study of the interrelation of energy with chemical reactions or with a physical change of state within the confines of the laws of thermodynamics. The primary objective of chemical thermodynamics is to determine the spontaneity of a given transformation.


=== Equilibrium thermodynamics ===
Equilibrium thermodynamics is the study of transfers of matter and energy in systems or bodies that, by agencies in their surroundings, can be driven from one state of thermodynamic equilibrium to another. The term 'thermodynamic equilibrium' indicates a state of balance, in which all macroscopic flows are zero; in the case of the simplest systems or bodies, their intensive properties are homogeneous, and their pressures are perpendicular to their boundaries. In an equilibrium state there are no unbalanced potentials, or driving forces, between macroscopically distinct parts of the system. A central aim in equilibrium thermodynamics is: given a system in a well-defined initial equilibrium state, and given its surroundings, and given its constitutive walls, to calculate what will be the final equilibrium state of the system after a specified thermodynamic operation has changed its walls or surroundings.


=== Non-equilibrium thermodynamics ===
Non-equilibrium thermodynamics is a branch of thermodynamics that deals with systems that are not in thermodynamic equilibrium. Most systems found in nature are not in thermodynamic equilibrium because they are not in stationary states, and are continuously and discontinuously subject to flux of matter and energy to and from other systems. The thermodynamic study of non-equilibrium systems requires more general concepts than are dealt with by equilibrium thermodynamics. Many natural systems still today remain beyond the scope of currently known macroscopic thermodynamic methods.


== Laws of thermodynamics ==

Thermodynamics is principally based on a set of four laws which are universally valid when applied to systems that fall within the constraints implied by each. In the various theoretical descriptions of thermodynamics these laws may be expressed in seemingly differing forms, but the most prominent formulations are the following.


=== Zeroth law ===
The zeroth law of thermodynamics states: If two systems are each in thermal equilibrium with a third, they are also in thermal equilibrium with each other.
This statement implies that thermal equilibrium is an equivalence relation on the set of thermodynamic systems under consideration. Systems are said to be in equilibrium if the small, random exchanges between them (e.g. Brownian motion) do not lead to a net change in energy. This law is tacitly assumed in every measurement of temperature. Thus, if one seeks to decide whether two bodies are at the same temperature, it is not necessary to bring them into contact and measure any changes of their observable properties in time. The law provides an empirical definition of temperature, and justification for the construction of practical thermometers.
The zeroth law was not initially recognized as a separate law of thermodynamics, as its basis in thermodynamical equilibrium was implied in the other laws. The first, second, and third laws had been explicitly stated already, and found common acceptance in the physics community before the importance of the zeroth law for the definition of temperature was realized. As it was impractical to renumber the other laws, it was named the zeroth law.


=== First law ===

The first law of thermodynamics states: In a process without transfer of matter, the change in internal energy, 
  
    
      
        Δ
        U
      
    
    {\displaystyle \Delta U}
  
, of a thermodynamic system is equal to the energy gained as heat, 
  
    
      
        Q
      
    
    {\displaystyle Q}
  
, less the thermodynamic work, 
  
    
      
        W
      
    
    {\displaystyle W}
  
, done by the system on its surroundings.

  
    
      
        Δ
        U
        =
        Q
        −
        W
      
    
    {\displaystyle \Delta U=Q-W}
  
.
where 
  
    
      
        Δ
        U
      
    
    {\displaystyle \Delta U}
  
 denotes the change in the internal energy of a closed system (for which heat or work through the system boundary are possible, but matter transfer is not possible), 
  
    
      
        Q
      
    
    {\displaystyle Q}
  
 denotes the quantity of energy supplied to the system as heat, and 
  
    
      
        W
      
    
    {\displaystyle W}
  
 denotes the amount of thermodynamic work done by the system on its surroundings. An equivalent statement is that perpetual motion machines of the first kind are impossible; work 
  
    
      
        W
      
    
    {\displaystyle W}
  
 done by a system on its surrounding requires that the system's internal energy 
  
    
      
        U
      
    
    {\displaystyle U}
  
 decrease or be consumed, so that the amount of internal energy lost by that work must be resupplied as heat 
  
    
      
        Q
      
    
    {\displaystyle Q}
  
 by an external energy source or as work by an external machine acting on the system (so that 
  
    
      
        U
      
    
    {\displaystyle U}
  
 is recovered) to make the system work continuously.
For processes that include transfer of matter, a further statement is needed: With due account of the respective fiducial reference states of the systems, when two systems, which may be of different chemical compositions, initially separated only by an impermeable wall, and otherwise isolated, are combined into a new system by the thermodynamic operation of removal of the wall, then

  
    
      
        
          U
          
            0
          
        
        =
        
          U
          
            1
          
        
        +
        
          U
          
            2
          
        
      
    
    {\displaystyle U_{0}=U_{1}+U_{2}}
  
,
where U0 denotes the internal energy of the combined system, and U1 and U2 denote the internal energies of the respective separated systems.
Adapted for thermodynamics, this law is an expression of the principle of conservation of energy, which states that energy can be transformed (changed from one form to another), but cannot be created or destroyed.
Internal energy is a principal property of the thermodynamic state, while heat and work are modes of energy transfer by which a process may change this state. A change of internal energy of a system may be achieved by any combination of heat added or removed and work performed on or by the system. As a function of state, the internal energy does not depend on the manner, or on the path through intermediate steps, by which the system arrived at its state.


=== Second law ===
A traditional version of the second law of thermodynamics states: Heat does not spontaneously flow from a colder body to a hotter body.
The second law refers to a system of matter and radiation, initially with inhomogeneities in temperature, pressure, chemical potential, and other intensive properties, that are due to internal 'constraints', or impermeable rigid walls, within it, or to externally imposed forces. The law observes that, when the system is isolated from the outside world and from those forces, there is a definite thermodynamic quantity, its entropy, that increases as the constraints are removed, eventually reaching a maximum value at thermodynamic equilibrium, when the inhomogeneities practically vanish. For systems that are initially far from thermodynamic equilibrium, though several have been proposed, there is known no general physical principle that determines the rates of approach to thermodynamic equilibrium, and thermodynamics does not deal with such rates. The many versions of the second law all express the general irreversibility of the transitions involved in systems approaching thermodynamic equilibrium.
In macroscopic thermodynamics, the second law is a basic observation applicable to any actual thermodynamic process; in statistical thermodynamics, the second law is postulated to be a consequence of molecular chaos.


=== Third law ===
The third law of thermodynamics states: As the temperature of a system approaches absolute zero, all processes cease and the entropy of the system approaches a minimum value.
This law of thermodynamics is a statistical law of nature regarding entropy and the impossibility of reaching absolute zero of temperature. This law provides an absolute reference point for the determination of entropy. The entropy determined relative to this point is the absolute entropy. Alternative definitions include "the entropy of all systems and of all states of a system is smallest at absolute zero," or equivalently "it is impossible to reach the absolute zero of temperature by any finite number of processes".
Absolute zero, at which all activity would stop if it were possible to achieve, is −273.15 °C (degrees Celsius), or −459.67 °F (degrees Fahrenheit), or 0 K (kelvin), or 0° R (degrees Rankine).


== System models ==

An important concept in thermodynamics is the thermodynamic system, which is a precisely defined region of the universe under study. Everything in the universe except the system is called the surroundings. A system is separated from the remainder of the universe by a boundary which may be a physical or notional, but serve to confine the system to a finite volume. Segments of the boundary are often described as walls; they have respective defined 'permeabilities'. Transfers of energy as work, or as heat, or of matter, between the system and the surroundings, take place through the walls, according to their respective permeabilities.
Matter or energy that pass across the boundary so as to effect a change in the internal energy of the system need to be accounted for in the energy balance equation. The volume contained by the walls can be the region surrounding a single atom resonating energy, such as Max Planck defined in 1900; it can be a body of steam or air in a steam engine, such as Sadi Carnot defined in 1824. The system could also be just one nuclide (i.e. a system of quarks) as hypothesized in quantum thermodynamics. When a looser viewpoint is adopted, and the requirement of thermodynamic equilibrium is dropped, the system can be the body of a tropical cyclone, such as Kerry Emanuel theorized in 1986 in the field of atmospheric thermodynamics, or the event horizon of a black hole.
Boundaries are of four types: fixed, movable, real, and imaginary. For example, in an engine, a fixed boundary means the piston is locked at its position, within which a constant volume process might occur. If the piston is allowed to move that boundary is movable while the cylinder and cylinder head boundaries are fixed. For closed systems, boundaries are real while for open systems boundaries are often imaginary. In the case of a jet engine, a fixed imaginary boundary might be assumed at the intake of the engine, fixed boundaries along the surface of the case and a second fixed imaginary boundary across the exhaust nozzle.
Generally, thermodynamics distinguishes three classes of systems, defined in terms of what is allowed to cross their boundaries:

As time passes in an isolated system, internal differences of pressures, densities, and temperatures tend to even out. A system in which all equalizing processes have gone to completion is said to be in a state of thermodynamic equilibrium.
Once in thermodynamic equilibrium, a system's properties are, by definition, unchanging in time. Systems in equilibrium are much simpler and easier to understand than are systems which are not in equilibrium. Often, when analysing a dynamic thermodynamic process, the simplifying assumption is made that each intermediate state in the process is at equilibrium, producing thermodynamic processes which develop so slowly as to allow each intermediate step to be an equilibrium state and are said to be reversible processes.


== States and processes ==
When a system is at equilibrium under a given set of conditions, it is said to be in a definite thermodynamic state. The state of the system can be described by a number of state quantities that do not depend on the process by which the system arrived at its state. They are called intensive variables or extensive variables according to how they change when the size of the system changes. The properties of the system can be described by an equation of state which specifies the relationship between these variables. State may be thought of as the instantaneous quantitative description of a system with a set number of variables held constant.
A thermodynamic process may be defined as the energetic evolution of a thermodynamic system proceeding from an initial state to a final state. It can be described by process quantities. Typically, each thermodynamic process is distinguished from other processes in energetic character according to what parameters, such as temperature, pressure, or volume, etc., are held fixed; Furthermore, it is useful to group these processes into pairs, in which each variable held constant is one member of a conjugate pair.
Several commonly studied thermodynamic processes are:

Adiabatic process: occurs without loss or gain of energy by heat
Isenthalpic process: occurs at a constant enthalpy
Isentropic process: a reversible adiabatic process, occurs at a constant entropy
Isobaric process: occurs at constant pressure
Isochoric process: occurs at constant volume (also called isometric/isovolumetric)
Isothermal process: occurs at a constant temperature
Steady-state process: occurs without a change in the internal energy


== Instrumentation ==
There are two types of thermodynamic instruments, the meter and the reservoir. A thermodynamic meter is any device which measures any parameter of a thermodynamic system. In some cases, the thermodynamic parameter is actually defined in terms of an idealized measuring instrument. For example, the zeroth law states that if two bodies are in thermal equilibrium with a third body, they are also in thermal equilibrium with each other. This principle, as noted by James Maxwell in 1872, asserts that it is possible to measure temperature. An idealized thermometer is a sample of an ideal gas at constant pressure. From the ideal gas law pV=nRT, the volume of such a sample can be used as an indicator of temperature; in this manner it defines temperature. Although pressure is defined mechanically, a pressure-measuring device, called a barometer may also be constructed from a sample of an ideal gas held at a constant temperature. A calorimeter is a device which is used to measure and define the internal energy of a system.
A thermodynamic reservoir is a system which is so large that its state parameters are not appreciably altered when it is brought into contact with the system of interest. When the reservoir is brought into contact with the system, the system is brought into equilibrium with the reservoir. For example, a pressure reservoir is a system at a particular pressure, which imposes that pressure upon the system to which it is mechanically connected. The Earth's atmosphere is often used as a pressure reservoir. The ocean can act as temperature reservoir when used to cool power plants.


== Conjugate variables ==

The central concept of thermodynamics is that of energy, the ability to do work. By the First Law, the total energy of a system and its surroundings is conserved. Energy may be transferred into a system by heating, compression, or addition of matter, and extracted from a system by cooling, expansion, or extraction of matter. In mechanics, for example, energy transfer equals the product of the force applied to a body and the resulting displacement.
Conjugate variables are pairs of thermodynamic concepts, with the first being akin to a "force" applied to some thermodynamic system, the second being akin to the resulting "displacement", and the product of the two equaling the amount of energy transferred. The common conjugate variables are:

Pressure–volume (the mechanical parameters);
Temperature–entropy (thermal parameters);
Chemical potential–particle number (material parameters).


== Potentials ==
Thermodynamic potentials are different quantitative measures of the stored energy in a system. Potentials are used to measure the energy changes in systems as they evolve from an initial state to a final state. The potential used depends on the constraints of the system, such as constant temperature or pressure. For example, the Helmholtz and Gibbs energies are the energies available in a system to do useful work when the temperature and volume or the pressure and temperature are fixed, respectively. Thermodynamic potentials cannot be measured in laboratories, but can be computed using molecular thermodynamics.
The five most well known potentials are:

where 
  
    
      
        T
      
    
    {\displaystyle T}
  
 is the temperature, 
  
    
      
        S
      
    
    {\displaystyle S}
  
 the entropy, 
  
    
      
        p
      
    
    {\displaystyle p}
  
 the pressure, 
  
    
      
        V
      
    
    {\displaystyle V}
  
 the volume, 
  
    
      
        μ
      
    
    {\displaystyle \mu }
  
 the chemical potential, 
  
    
      
        N
      
    
    {\displaystyle N}
  
 the number of particles in the system, and 
  
    
      
        i
      
    
    {\displaystyle i}
  
 is the count of particles types in the system.
Thermodynamic potentials can be derived from the energy balance equation applied to a thermodynamic system. Other thermodynamic potentials can also be obtained through Legendre transformation.


== Axiomatic thermodynamics ==
Axiomatic thermodynamics is a mathematical discipline that aims to describe thermodynamics in terms of rigorous axioms, for example by finding a mathematically rigorous way to express the familiar laws of thermodynamics.
The first attempt at an axiomatic theory of thermodynamics was Constantin Carathéodory's 1909 work Investigations on the Foundations of Thermodynamics, which made use of Pfaffian systems and the concept of adiabatic accessibility, a notion that was introduced by Carathéodory himself. In this formulation, thermodynamic concepts such as heat, entropy, and temperature are derived from quantities that are more directly measurable. Theories that came after, differed in the sense that they made assumptions regarding thermodynamic processes with arbitrary initial and final states, as opposed to considering only neighboring states.


== Applied fields ==


== See also ==

Thermodynamic process path
Carnot engine (intuitive explanation)


=== Lists and timelines ===
List of important publications in thermodynamics
List of textbooks on thermodynamics and statistical mechanics
List of thermal conductivities
List of thermodynamic properties
Table of thermodynamic equations
Timeline of thermodynamics
Thermodynamic equations


== Notes ==


== References ==


== Further reading ==
Goldstein, Martin & Inge F. (1993). The Refrigerator and the Universe. Harvard University Press. ISBN 978-0-674-75325-9. OCLC 32826343. A nontechnical introduction, good on historical and interpretive matters.
Kazakov, Andrei; Muzny, Chris D.; Chirico, Robert D.; Diky, Vladimir V.; Frenkel, Michael (2008). "Web Thermo Tables – an On-Line Version of the TRC Thermodynamic Tables". Journal of Research of the National Institute of Standards and Technology. 113 (4): 209–220. doi:10.6028/jres.113.016. ISSN 1044-677X. PMC 4651616. PMID 27096122.
Gibbs J.W. (1928). The Collected Works of J. Willard Gibbs Thermodynamics. New York: Longmans, Green and Co. Vol. 1, pp. 55–349.
Guggenheim E.A. (1933). Modern thermodynamics by the methods of Willard Gibbs. London: Methuen & co. ltd.
Denbigh K. (1981). The Principles of Chemical Equilibrium: With Applications in Chemistry and Chemical Engineering. London: Cambridge University Press.
Stull, D.R., Westrum Jr., E.F. and Sinke, G.C. (1969). The Chemical Thermodynamics of Organic Compounds. London: John Wiley and Sons, Inc.{{cite book}}:  CS1 maint: multiple names: authors list (link)
Bazarov I.P. (2010). Thermodynamics: Textbook. St. Petersburg: Lan publishing house. p. 384. ISBN 978-5-8114-1003-3. 5th ed. (in Russian)
Bawendi Moungi G., Alberty Robert A. and Silbey Robert J. (2004). Physical Chemistry. J. Wiley & Sons, Incorporated.
Alberty Robert A. (2003). Thermodynamics of Biochemical Reactions. Wiley-Interscience.
Alberty Robert A. (2006). Biochemical Thermodynamics: Applications of Mathematica. Vol. 48. John Wiley & Sons, Inc. pp. 1–458. ISBN 978-0-471-75798-6. PMID 16878778. {{cite book}}: |journal= ignored (help)
Dill Ken A., Bromberg Sarina (2011). Molecular Driving Forces: Statistical Thermodynamics in Biology, Chemistry, Physics, and Nanoscience. Garland Science. ISBN 978-0-8153-4430-8.
M. Scott Shell (2015). Thermodynamics and Statistical Mechanics: An Integrated Approach. Cambridge University Press. ISBN 978-1107656789.
Douglas E. Barrick (2018). Biomolecular Thermodynamics: From Theory to Applications. CRC Press. ISBN 978-1-4398-0019-5.
The following titles are more technical:

Bejan, Adrian (2016). Advanced Engineering Thermodynamics (4 ed.). Wiley. ISBN 978-1-119-05209-8.
Cengel, Yunus A., & Boles, Michael A. (2002). Thermodynamics – an Engineering Approach. McGraw Hill. ISBN 978-0-07-238332-4. OCLC 45791449.{{cite book}}:  CS1 maint: multiple names: authors list (link)
Dunning-Davies, Jeremy (1997). Concise Thermodynamics: Principles and Applications. Horwood Publishing. ISBN 978-1-8985-6315-0. OCLC 36025958.
Kroemer, Herbert & Kittel, Charles (1980). Thermal Physics. W.H. Freeman Company. ISBN 978-0-7167-1088-2. OCLC 32932988.


== External links ==
 Media related to Thermodynamics at Wikimedia Commons

Callendar, Hugh Longbourne (1911). "Thermodynamics" . Encyclopædia Britannica. Vol. 26 (11th ed.). pp. 808–814.
Thermodynamics Data & Property Calculation Websites Archived 11 February 2014 at the Wayback Machine
Thermodynamics Educational Websites Archived 14 June 2015 at the Wayback Machine
Biochemistry Thermodynamics
Thermodynamics and Statistical Mechanics
Engineering Thermodynamics – A Graphical Approach
Thermodynamics and Statistical Mechanics by Richard Fitzpatrick

Computer programming or coding is the composition of sequences of instructions, called programs, that computers can follow to perform tasks. It involves designing and implementing algorithms, step-by-step specifications of procedures, by writing code in one or more programming languages. Programmers typically use high-level programming languages that are more easily intelligible to humans than machine code, which is directly executed by the central processing unit. Proficient programming usually requires expertise in several different subjects, including knowledge of the application domain, details of programming languages and generic code libraries, specialized algorithms, and formal logic.
Auxiliary tasks accompanying and related to programming include analyzing requirements, testing, debugging (investigating and fixing problems), implementation of build systems, and management of derived artifacts, such as programs' machine code. While these are sometimes considered programming, often the term software development is used for this larger overall process – with the terms programming, implementation, and coding reserved for the writing and editing of code per se. Sometimes software development is known as software engineering, especially when it employs formal methods or follows an engineering design process.


== History ==

Programmable devices have existed for centuries. As early as the 9th century, a programmable music sequencer was invented by the Persian Banu Musa brothers, who described an automated mechanical flute player in the Book of Ingenious Devices. In 1206, the Arab engineer Al-Jazari invented a programmable drum machine where a musical mechanical automaton could be made to play different rhythms and drum patterns, via pegs and cams. In 1801, the Jacquard loom could produce entirely different weaves by changing the "program" – a series of pasteboard cards with holes punched in them.
Code-breaking algorithms have also existed for centuries. In the 9th century, the Arab mathematician Al-Kindi described a cryptographic algorithm for deciphering encrypted code, in A Manuscript on Deciphering Cryptographic Messages. He gave the first description of cryptanalysis by frequency analysis, the earliest code-breaking algorithm.
The first computer program is generally dated to 1843 when mathematician Ada Lovelace published an algorithm to calculate a sequence of Bernoulli numbers, intended to be carried out by Charles Babbage's Analytical Engine. The algorithm, which was conveyed through notes on a translation of Luigi Federico Menabrea's paper on the analytical engine was mainly conceived by Lovelace as can be discerned through her correspondence with Babbage. However, Charles Babbage himself had written a program for the AE in 1837. Lovelace was also the first to see a broader application for the analytical engine beyond mathematical calculations.

In the 1880s, Herman Hollerith invented the concept of storing data in machine-readable form. Later a control panel (plug board) added to his 1906 Type I Tabulator allowed it to be programmed for different jobs, and by the late 1940s, unit record equipment such as the IBM 602 and IBM 604, were programmed by control panels in a similar way, as were the first electronic computers. However, with the concept of the stored-program computer introduced in 1949, both programs and data were stored and manipulated in the same way in computer memory. Hands-on programming courses that integrate hardware and software have been shown to improve retention and engagement among first-year engineering students.


=== Machine language ===
Machine code was the language of early programs, written in the instruction set of the particular machine, often in binary notation. Soon, assembly languages were developed, allowing programmers to write instructions in a textual format (e.g., ADD X, TOTAL), using abbreviations for operation codes and meaningful names for memory addresses. However, because an assembly language is little more than a different notation for a machine language,  two machines with different instruction sets also have different assembly languages.


=== Compiler languages ===

High-level languages made the process of developing a program simpler and more understandable, and less bound to the underlying hardware. 
The first compiler related tool, the A-0 System, was developed in 1952 by Grace Hopper, who also coined the term 'compiler'. FORTRAN, the first widely used high-level language to have a functional implementation, came out in 1957, and many other languages were soon developed—in particular, COBOL aimed at commercial data processing, and Lisp for computer research.
These compiled languages allow the programmer to write programs in terms that are syntactically richer, and more capable of abstracting the code, making it easy to target varying machine instruction sets via compilation declarations and heuristics. Compilers harnessed the power of computers to make programming easier by allowing programmers to specify calculations by entering a formula using infix notation.


=== Source code entry ===

Programs were mostly entered using punched cards or paper tape. By the late 1960s, data storage devices and computer terminals became inexpensive enough that programs could be created by typing directly into the computers. Text editors were also developed that allowed changes and corrections to be made much more easily than with punched cards.


== Modern programming ==


=== Quality requirements ===

Whatever the approach to development may be, the final program must satisfy some fundamental properties. The following properties are among the most important:

Reliability: how often the results of a program are correct. This depends on conceptual correctness of algorithms and minimization of programming mistakes, such as mistakes in resource management (e.g., buffer overflows and race conditions) and logic errors (such as division by zero or off-by-one errors).
Robustness: how well a program anticipates problems due to errors (not bugs). This includes situations such as incorrect, inappropriate or corrupt data, unavailability of needed resources such as memory, operating system services, and network connections, user error, and unexpected power outages.
Usability: the ergonomics of a program: the ease with which a person can use the program for its intended purpose or in some cases even unanticipated purposes. Such issues can make or break its success even regardless of other issues. This involves a wide range of textual, graphical, and sometimes hardware elements that improve the clarity, intuitiveness, cohesiveness, and completeness of a program's user interface.
Portability: the range of computer hardware and operating system platforms on which the source code of a program can be compiled/interpreted and run. This depends on differences in the programming facilities provided by the different platforms, including hardware and operating system resources, expected behavior of the hardware and operating system, and availability of platform-specific compilers (and sometimes libraries) for the language of the source code.
Maintainability: the ease with which a program can be modified by its present or future developers in order to make improvements or to customize, fix bugs and security holes, or adapt it to new environments. Good practices during initial development make the difference in this regard. This quality may not be directly apparent to the end user but it can significantly affect the fate of a program over the long term.
Efficiency/performance: Measure of system resources a program consumes (processor time, memory space, slow devices such as disks, network bandwidth and to some extent even user interaction): the less, the better. This also includes careful management of resources, for example cleaning up temporary files and eliminating memory leaks. This is often discussed under the shadow of a chosen programming language. Although the language certainly affects performance, even slower languages, such as Python, can execute programs instantly from a human perspective. Speed, resource usage, and performance are important for programs that bottleneck the system, but efficient use of programmer time is also important and is related to cost: more hardware may be cheaper.
Using automated tests and fitness functions can help to maintain some of the aforementioned attributes.


=== Readability of source code ===
In computer programming, readability refers to the ease with which a human reader can comprehend the purpose, control flow, and operation of source code. It affects the aspects of quality above, including portability, usability and most importantly maintainability.

Readability is important because programmers spend the majority of their time reading, trying to understand, reusing, and modifying existing source code, rather than writing new source code. Unreadable code often leads to bugs, inefficiencies, and duplicated code. A study found that a few simple readability transformations made code shorter and drastically reduced the time to understand it.
Following a consistent programming style often helps readability. However, readability is more than just programming style. Many factors, having little or nothing to do with the ability of the computer to efficiently compile and execute the code, contribute to readability. Some of these factors include:

Different indent styles (whitespace)
Comments and documentation
Decomposition
Naming conventions for objects (such as variables, classes, functions, procedures, etc.)
The presentation aspects of this (such as indents, line breaks, color highlighting, and so on) are often handled by the source code editor, but the content aspects reflect the programmer's talent and skills.
Various visual programming languages have also been developed with the intent to resolve readability concerns by adopting non-traditional approaches to code structure and display. Integrated development environments (IDEs) aim to integrate all such help. Techniques like Code refactoring can enhance readability.


=== Algorithmic complexity ===
The academic field and the engineering practice of computer programming are concerned with discovering and implementing the most efficient algorithms for a given class of problems. For this purpose, algorithms are classified into orders using Big O notation, which expresses resource use—such as execution time or memory consumption—in terms of the size of an input. Expert programmers are familiar with a variety of well-established algorithms and their respective complexities and use this knowledge to choose algorithms that are best suited to the circumstances.


=== Methodologies ===
The first step in most formal software development processes is requirements analysis, followed by testing to determine value modeling, implementation, and failure elimination (debugging). There exist a lot of different approaches for each of those tasks. One approach popular for requirements analysis is Use Case analysis. Many programmers use forms of Agile software development where the various stages of formal software development are more integrated together into short cycles that take a few weeks rather than years. There are many approaches to the Software development process.
Popular modeling techniques include Object-Oriented Analysis and Design (OOAD) and Model-Driven Architecture (MDA). The Unified Modeling Language (UML) is a notation used for both the OOAD and MDA.
A similar technique used for database design is Entity-Relationship Modeling (ER Modeling).
Implementation techniques include imperative languages (object-oriented or procedural), functional languages, and logic programming languages.


=== Measuring language usage ===
It is very difficult to determine what are the most popular modern programming languages. Methods of measuring programming language popularity include: counting the number of job advertisements that mention the language, the number of books sold and courses teaching the language (this overestimates the importance of newer languages), and estimates of the number of existing lines of code written in the language (this underestimates the number of users of business languages such as COBOL).
Some languages are popular for writing particular kinds of applications, while other languages are used to write many different kinds of applications. For example, COBOL is still prevalent in corporate data centers often on large mainframe computers, Fortran in engineering applications, scripting languages in Web development, and C in embedded software. Many applications use a mix of several languages in their construction and use. New languages are generally designed around the syntax of a prior language with new functionality added, (for example C++ adds object-orientation to C, and Java adds memory management and bytecode to C++, but as a result, loses efficiency and the ability for low-level manipulation).


=== Debugging ===

Debugging is a very important task in the software development process since having defects in a program can have significant consequences for its users. Some languages are more prone to some kinds of faults because their specification does not require compilers to perform as much checking as other languages. Use of a static code analysis tool can help detect some possible problems. Normally the first step in debugging is to attempt to reproduce the problem. This can be a non-trivial task, for example as with parallel processes or some unusual software bugs. Also, specific user environment and usage history can make it difficult to reproduce the problem.
After the bug is reproduced, the input of the program may need to be simplified to make it easier to debug. For example, when a bug in a compiler can make it crash when parsing some large source file, a simplification of the test case that results in only few lines from the original source file can be sufficient to reproduce the same crash. Trial-and-error/divide-and-conquer is needed: the programmer will try to remove some parts of the original test case and check if the problem still exists. When debugging the problem in a GUI, the programmer can try to skip some user interaction from the original problem description and check if the remaining actions are sufficient for bugs to appear. Scripting and breakpointing are also part of this process.
Debugging is often done with IDEs. Standalone debuggers like GDB are also used, and these often provide less of a visual environment, usually using a command line. Some text editors such as Emacs allow GDB to be invoked through them, to provide a visual environment. One study found that teaching structured languages such as Pascal improved novice programmers’ debugging accuracy and overall comprehension.


== Programming languages ==

Different programming languages support different styles of programming (called programming paradigms). The choice of language used is subject to many considerations, such as company policy, suitability to task, availability of third-party packages, or individual preference. Ideally, the programming language best suited for the task at hand will be selected. Trade-offs from this ideal involve finding enough programmers who know the language to build a team, the availability of compilers for that language, and the efficiency with which programs written in a given language execute. Languages form an approximate spectrum from "low-level" to "high-level"; "low-level" languages are typically more machine-oriented and faster to execute, whereas "high-level" languages are more abstract and easier to use but execute less quickly. It is usually easier to code in "high-level" languages than in "low-level" ones.
Programming languages are essential for software development. They are the building blocks for all software, from the simplest applications to the most sophisticated ones.
Allen Downey, in his book How To Think Like A Computer Scientist, writes:

The details look different in different languages, but a few basic instructions appear in just about every language:
Input: Gather data from the keyboard, a file, or some other device.
Output: Display data on the screen or send data to a file or other device.
Arithmetic: Perform basic arithmetical operations like addition and multiplication.
Conditional Execution: Check for certain conditions and execute the appropriate sequence of statements.
Repetition: Perform some action repeatedly, usually with some variation.
Many computer languages provide a mechanism to call functions provided by shared libraries. Provided the functions in a library follow the appropriate run-time conventions (e.g., method of passing arguments), then these functions may be written in any other language. Annual rankings by IEEE Spectrum analyze programming language popularity using metrics such as job postings, search trends, and developer activity.


== Learning to program ==

Learning to program has a long history related to professional standards and practices, academic initiatives and curriculum, and commercial books and materials for students, self-taught learners, hobbyists, and others who desire to create or customize software for personal use. Since the 1960s, learning to program has taken on the characteristics of a popular movement, with the rise of academic disciplines, inspirational leaders, collective identities, and strategies to grow the movement and make institutionalize change. Through these social ideals and educational agendas, learning to code has become important not just for scientists and engineers, but for millions of citizens who have come to believe that creating software is beneficial to society and its members. Computer science education participation has expanded significantly in recent years, with millions of students gaining introductory programming experience through schools and online platforms.


=== Context ===
In 1957, there were approximately 15,000 computer programmers employed in the U.S., a figure that accounts for 80% of the world's active developers. In 2014, there were approximately 18.5 million professional programmers in the world, of which 11 million can be considered professional and 7.5 million student or hobbyists. Before the rise of the commercial Internet in the mid-1990s, most programmers learned about software construction through books, magazines, user groups, and informal instruction methods, with academic coursework and corporate training playing important roles for professional workers.
The first book containing specific instructions about how to program a computer may have been Maurice Wilkes, David Wheeler, and Stanley Gill's Preparation of Programs for an Electronic Digital Computer (1951). The book offered a selection of common subroutines for handling basic operations on the EDSAC, one of the world's first stored-program computers.
When high-level languages arrived, they were introduced by numerous books and materials that explained language keywords, managing program flow, working with data, and other concepts. These languages included FLOW-MATIC, COBOL, FORTRAN, ALGOL, Pascal, BASIC, and C. An example of an early programming primer from these years is Marshal H. Wrubel's A Primer of Programming for Digital Computers (1959), which included step-by-step instructions for filling out coding sheets, creating punched cards, and using the keywords in IBM's early FORTRAN system. Daniel McCracken's A Guide to FORTRAN Programming (1961) presented FORTRAN to a larger audience, including students and office workers.
In 1961, Alan Perlis suggested that all university freshmen at Carnegie Technical Institute take a course in computer programming. His advice was published in the popular technical journal Computers and Automation, which became a regular source of information for professional programmers.
Programmers soon had a range of learning texts at their disposal. Programmer's references listed keywords and functions related to a language, often in alphabetical order, as well as technical information about compilers and related systems. An early example was IBM's Programmers' Reference Manual: the FORTRAN Automatic Coding System for the IBM 704 EDPM (1956).
Over time, the genre of programmer's guides emerged, which presented the features of a language in tutorial or step by step format. Many early primers started with a program known as "Hello, World", which presented the shortest program a developer could create in a given system. Programmer's guides then went on to discuss core topics like declaring variables, data types, formulas, flow control, user-defined functions, manipulating data, and other topics.
Early and influential programmer's guides included John G. Kemeny and Thomas E. Kurtz's BASIC Programming (1967), Kathleen Jensen and Niklaus Wirth's The Pascal User Manual and Report (1971), and Brian W. Kernighan and Dennis Ritchie's The C Programming Language (1978). Similar books for popular audiences (but with a much lighter tone) included Bob Albrecht's My Computer Loves Me When I Speak BASIC (1972), Al Kelley and Ira Pohl's A Book on C (1984), and Dan Gookin's C for Dummies (1994).
Beyond language-specific primers, there were numerous books and academic journals that introduced professional programming practices. Many were designed for university courses in computer science, software engineering, or related disciplines. Donald Knuth's The Art of Computer Programming (1968 and later), presented hundreds of computational algorithms and their analysis. The Elements of Programming Style (1974), by Brian W. Kernighan and P. J. Plauger, concerned itself with programming style, the idea that programs should be written not only to satisfy the compiler but human readers. Jon Bentley's Programming Pearls (1986) offered practical advice about the art and craft of programming in professional and academic contexts. Texts specifically designed for students included Doug Cooper and Michael Clancy's Oh Pascal! (1982), Alfred Aho's Data Structures and Algorithms (1983), and Daniel Watt's Learning with Logo (1983).


=== Technical publishers ===
As personal computers became mass-market products, thousands of trade books and magazines sought to teach professional, hobbyist, and casual users to write computer programs. A sample of these learning resources includes BASIC Computer Games, Microcomputer Edition (1978), by David Ahl; Programming the Z80 (1979), by Rodnay Zaks; Programmer's CP/M Handbook (1983), by Andy Johnson-Laird; C Primer Plus (1984), by Mitchell Waite and The Waite Group; The Peter Norton Programmer's Guide to the IBM PC (1985), by Peter Norton; Advanced MS-DOS (1986), by Ray Duncan; Learn BASIC Now (1989), by Michael Halvorson and David Rygymr; Programming Windows (1992 and later), by Charles Petzold; Code Complete: A Practical Handbook for Software Construction (1993), by Steve McConnell; and Tricks of the Game-Programming Gurus (1994), by André LaMothe.
The PC software industry spurred the creation of numerous book publishers that offered programming primers and tutorials, as well as books for advanced software developers. These publishers included Addison-Wesley, IDG, Macmillan Inc., McGraw-Hill, Microsoft Press, O'Reilly Media, Prentice Hall, Sybex, Ventana Press, Waite Group Press, Wiley, Wrox Press, and Ziff-Davis.
Computer magazines and journals also provided learning content for professional and hobbyist programmers. A partial list of these resources includes Amiga World, Byte (magazine), Communications of the ACM, Computer (magazine), Compute!, Computer Language (magazine), Computers and Electronics, Dr. Dobb's Journal, IEEE Software, Macworld, PC Magazine, PC/Computing, and UnixWorld.


=== Digital learning / online resources ===

Between 2000 and 2010, computer book and magazine publishers declined significantly as providers of programming instruction, as programmers moved to Internet resources to expand their access to information. This shift brought forward new digital products and mechanisms to learn programming skills. During the transition, digital books from publishers transferred information that had traditionally been delivered in print to new and expanding audiences.
Important Internet resources for learning to code included blogs, books, wikis, videos, online databases, journals, subscription sites, conference papers. and custom websites focused on coding skills. In recent years, platforms like LeetCode, HackerRank, and freeCodeCamp have become popular for learning programming, practicing coding challenges, and preparing for technical interviews. New commercial resources included YouTube videos, Lynda.com tutorials (later LinkedIn Learning), Khan Academy, Codecademy, GitHub, W3Schools, Codewars, and numerous coding bootcamps.
Most software development systems and game engines included rich online help resources, including integrated development environments (IDEs), context-sensitive help, APIs, and other digital resources. Commercial software development kits (SDKs) also provided a collection of software development tools and documentation in one installable package.
Commercial and non-profit organizations published learning websites for developers, created blogs, and established news feeds and social media resources about programming. Corporations like Apple, Microsoft, Oracle, Google, and Amazon built corporate websites providing support for programmers, including resources like the Microsoft Developer Network (MSDN). Contemporary movements like Hour of Code (Code.org) show how learning to program has become associated with digital learning strategies, education agendas, and corporate philanthropy.


== Programmers ==

Computer programmers are those who write computer software. Their jobs usually involve:

Although programming has been presented in the media as a somewhat mathematical subject, some research shows that good programmers have strong skills in natural human languages, and that learning to code is similar to learning a foreign language.


== See also ==

Code smell
Computer networking
Competitive programming
List of programmers
List of software programming journals
List of free and open-source software packages for programming
Lists of programming software development tools
Programming best practices
Systems programming
Wikibooks computer programming resources


== References ==


=== Sources ===
Ceruzzi, Paul E. (1998). History of Computing. Cambridge, Massachusetts: MIT Press. ISBN 9780262032551 – via EBSCOhost.
Evans, Claire L. (2018). Broad Band: The Untold Story of the Women Who Made the Internet. New York: Portfolio/Penguin. ISBN 9780735211759.
Gürer, Denise (1995). "Pioneering Women in Computer Science" (PDF). Communications of the ACM. 38 (1): 45–54. doi:10.1145/204865.204875. S2CID 6626310. Archived (PDF) from the original on October 9, 2022.
Smith, Erika E. (2013). "Recognizing a Collective Inheritance through the History of Women in Computing". CLCWeb: Comparative Literature and Culture. 15 (1): 1–9. doi:10.7771/1481-4374.1972.
Essinger, J., & EBSCO Publishing (Firm). (2014). Ada's algorithm: How lord byron's daughter ada lovelace launched the digital age. Melville House.


== Further reading ==
A.K. Hartmann, Practical Guide to Computer Simulations, Singapore: World Scientific (2009)
A. Hunt, D. Thomas, and W. Cunningham, The Pragmatic Programmer. From Journeyman to Master, Amsterdam: Addison-Wesley Longman (1999)
Brian W. Kernighan, The Practice of Programming, Pearson (1999)
Weinberg, Gerald M., The Psychology of Computer Programming, New York: Van Nostrand Reinhold (1971)
Edsger W. Dijkstra, A Discipline of Programming, Prentice-Hall (1976)
O.-J. Dahl, E.W.Dijkstra, C.A.R. Hoare, Structured Programming, Academic Press (1972)
David Gries, The Science of Programming, Springer-Verlag (1981)


== External links ==

 Media related to Computer programming at Wikimedia Commons
 Quotations related to Programming at Wikiquote

Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from  data and generalize to unseen data, and thus perform tasks without being explicitly programmed. Advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance.
Statistics and mathematical optimisation methods compose the foundations of machine learning. Data mining is a related field of study, focusing on exploratory data analysis (EDA) through unsupervised learning.
From a theoretical viewpoint, probably approximately correct learning provides a mathematical and statistical framework for describing machine learning. Most traditional machine learning and deep learning algorithms can be described as empirical risk minimisation under this framework.


== History ==

The term machine learning was coined in 1959 by Arthur Samuel, an IBM employee and pioneer in the field of computer gaming and artificial intelligence. The synonym self-teaching computers was also used during this time period.
The earliest machine learning program was introduced in the 1950s, when Samuel invented a computer program that calculated the chance of winning in checkers for each side, but the history of machine learning is rooted in decades of efforts to study human cognitive processes. In 1949, Canadian psychologist Donald Hebb published the book The Organization of Behavior, in which he introduced a theoretical neural structure formed by certain interactions among nerve cells. The Hebbian theory of neuron interaction set the groundwork for how many machine learning algorithms work, with connected artificial neurons changing the strength of their connections based on data. Other researchers who have studied human cognitive systems contributed to the modern machine learning technologies as well, including  Walter Pitts and Warren McCulloch, who proposed the first mathematical model of neural networks including algorithms that mirror human thought processes.
By the early 1960s, an experimental "learning machine" with punched tape memory, called Cybertron, had been developed by Raytheon Company to analyse sonar signals, electrocardiograms, and speech patterns using rudimentary reinforcement learning. It was repetitively "trained" by a human operator/teacher to recognise patterns and equipped with a "goof" button to cause it to reevaluate incorrect decisions. A representative book on research into machine learning during the 1960s was Nils Nilsson's book "Learning Machines", dealing mostly with machine learning for pattern classification. Interest related to pattern recognition continued into the 1970s, as described by Duda and Hart in 1973. In 1981, a report was given on using teaching strategies so that an artificial neural network learns to recognise 40 characters (26 letters, 10 digits, and 4 special symbols) from a computer terminal.
Tom M. Mitchell provided a widely quoted, more formal definition of the algorithms studied in the machine learning field: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P,  improves with experience E." This definition of the tasks in which machine learning is concerned is fundamentally operational rather than defining the field in cognitive terms. This follows Alan Turing's proposal in his paper "Computing Machinery and Intelligence", in which the question, "Can machines think?", is replaced by asking whether machines can convincingly imitate a human in its responses to human-posed questions.
In 2014, Ian Goodfellow and others introduced generative adversarial networks (GANs) which could produce realistic synthetic data. By 2016, AlphaGo had won against top human players in Go using reinforcement learning techniques.


== Relationships to other fields ==


=== Artificial intelligence ===

As a scientific endeavour, machine learning grew out of the quest for artificial intelligence (AI). In the early days of AI as an academic discipline, some researchers were interested in having machines learn from data. They attempted to approach the problem with various symbolic methods, as well as what were then termed "neural networks"; these were mostly perceptrons and other models that were later found to be reinventions of the generalised linear models of statistics. Probabilistic reasoning was also employed, especially in automated medical diagnosis.
However, an increasing emphasis on the logical, knowledge-based approach caused a rift between AI and machine learning. Probabilistic systems were plagued by theoretical and practical problems of data acquisition and representation. By 1980, expert systems had come to dominate AI, and statistics was out of favour. Work on symbolic/knowledge-based learning continued within AI, leading to inductive logic programming (ILP), but the more statistical line of research was now outside the field of AI proper, in pattern recognition and information retrieval. Neural network research was abandoned by AI and computer science around the same time. This subfield, termed "connectionism", was continued by researchers from other disciplines, including John Hopfield, David Rumelhart, and Geoffrey Hinton. Their main success came in the mid-1980s with the reinvention of backpropagation.
Machine learning (ML), reorganised and recognised as its own field, started to flourish in the 1990s. The field changed its goal from achieving artificial intelligence to tackling solvable problems of a practical nature. It shifted focus away from the symbolic approaches it had inherited from AI, and toward methods and models borrowed from statistics, fuzzy logic, and probability theory.


=== Data compression ===


=== Data mining ===
Machine learning and data mining often employ the same methods and overlap significantly, but while machine learning focuses on prediction based on known properties learned from the training data, data mining focuses on the discovery of previously unknown properties in the data (this is the analysis step of knowledge discovery in databases). Data mining uses many machine learning methods, but with different goals; on the other hand, machine learning also employs data mining methods as "unsupervised learning" or as a preprocessing step to improve learner accuracy. Much of the confusion between these two research communities comes from the basic assumptions they work with: in machine learning, performance is usually evaluated with respect to the ability to reproduce known knowledge, while in knowledge discovery and data mining (KDD) the key task is the discovery of previously unknown knowledge. Evaluated with respect to known knowledge, an uninformed (unsupervised) method will easily be outperformed by other supervised methods, while in a typical KDD task, supervised methods cannot be used due to the unavailability of training data.
Machine learning also has intimate ties to optimization: Many learning problems are formulated as minimisation of some loss function on a training set of examples. Loss functions express the discrepancy between the predictions of the model being trained and the actual problem instances (for example, in classification, one wants to assign a label to instances, and models are trained to correctly predict the preassigned labels of a set of examples).


=== Generalization ===
Characterizing the generalisation of various learning algorithms is an active topic of current research, especially for deep learning algorithms.


=== Statistics ===
Machine learning and statistics are closely related fields in terms of methods, but distinct in their principal goal: statistics draws population inferences from a sample, while machine learning finds generalisable predictive patterns.
Conventional statistical analyses require the a priori selection of a model most suitable for the study data set. In addition, only significant or theoretically relevant variables based on previous experience are included for analysis. In contrast, machine learning is not built on a pre-structured model; rather, the data shape the model by detecting underlying patterns. The more variables (input) used to train the model, the more accurate the ultimate model will be.
Leo Breiman distinguished two statistical modelling paradigms: the data model and the algorithmic model, wherein "algorithmic model" means more or less the machine learning algorithms like Random forest.
Some statisticians have adopted methods from machine learning, producing the field of statistical learning.


=== Statistical physics ===
Analytical and computational techniques derived from deep-rooted physics of disordered systems can be extended to large-scale problems, including machine learning, e.g., to analyse the weight space of deep neural networks. Statistical physics is thus finding applications in the area of medical diagnostics.


== Theory ==

A core objective of a learner is to generalise from its experience. Generalisation in this context is the ability of a learning machine to perform accurately on new, unseen examples/tasks after having experienced a learning data set. The training examples come from some generally unknown probability distribution (considered representative of the space of occurrences) and the learner has to build a general model about this space that enables it to produce sufficiently accurate predictions in new cases.
The computational analysis of machine learning algorithms and their performance is a branch of theoretical computer science known as computational learning theory. One major framework is the probably approximately correct learning model. Because training sets are finite and the future is uncertain, learning theory usually does not yield guarantees of the performance of algorithms. Instead, probabilistic bounds on the performance are quite common. The bias–variance decomposition is one way to quantify generalisation error.
For the best performance in the context of generalisation, the complexity of the hypothesis should match the complexity of the function underlying the data. If the hypothesis is less complex than the function, then the model has underfitted the data. If the complexity of the model is increased in response, then the training error decreases. But if the hypothesis is too complex, then the model is subject to overfitting and generalisation will be poorer.
In addition to performance bounds, learning theorists study the time complexity and feasibility of learning. In computational learning theory, a computation is considered feasible if it can be done in polynomial time. There are two kinds of time complexity results: Positive results show that a certain class of functions can be learned in polynomial time. Negative results show that certain classes cannot be learned in polynomial time.


== Approaches ==

Machine learning approaches are traditionally divided into three broad categories, which correspond to learning paradigms, depending on the nature of the "signal" or "feedback" available to the learning system:

Supervised learning: The computer is presented with example inputs and their desired outputs, given by a "teacher", and the goal is to learn a general rule that maps inputs to outputs.
Unsupervised learning: No labels are given to the learning algorithm, leaving it on its own to find structure in its input. Unsupervised learning can be a goal in itself (discovering hidden patterns in data) or a means towards an end (feature learning).
Reinforcement learning: A computer program interacts with a dynamic environment in which it must perform a certain goal (such as driving a vehicle or playing a game against an opponent). As it navigates its problem space, the program is provided rewards as feedback, which it tries to maximize, thus resulting in the program learning from experience.
Although each algorithm has advantages and limitations, no single algorithm works for all problems.


=== Supervised learning ===

Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs. The data, known as training data, consists of a set of training examples. Each training example has one or more inputs and the desired output, also known as a supervisory signal. In the mathematical model, each training example is represented by an array or vector, sometimes called a feature vector, and the training data is represented by a matrix. Through iterative optimisation of an objective function, supervised learning algorithms learn a function that can be used to predict the output associated with new inputs. An optimal function allows the algorithm to correctly determine the output for inputs that were not a part of the training data. An algorithm that improves the accuracy of its outputs or predictions over time is said to have learned to perform that task.
Types of supervised-learning algorithms include active learning, classification and regression. Classification algorithms are used when the outputs are restricted to a limited set of values, while regression algorithms are used when the outputs can take any numerical value within a range. For example, in a classification algorithm that filters emails, the input is an incoming email, and the output is the folder in which to file the email. In contrast, regression is used for tasks such as predicting a person's height based on factors like age and genetics or forecasting future temperatures based on historical data.
Similarity learning is an area of supervised machine learning closely related to regression and classification, but the goal is to learn from examples using a similarity function that measures how similar or related two objects are. It has applications in ranking, recommendation systems, visual identity tracking, face verification, and speaker verification.


=== Unsupervised learning ===

Unsupervised learning algorithms find structures in data that has not been labelled, classified or categorised. Instead of responding to feedback, unsupervised learning algorithms identify commonalities in the data and react based on the presence or absence of such commonalities in each new piece of data. Central applications of unsupervised machine learning include clustering, dimensionality reduction, and density estimation.
Cluster analysis is the assignment of a set of observations into subsets (called clusters) so that observations within the same cluster are similar according to one or more predesignated criteria, while observations drawn from different clusters are dissimilar. Different clustering techniques make different assumptions on the structure of the data, often defined by some similarity metric and evaluated, for example, by internal compactness, or the similarity between members of the same cluster, and separation, the difference between clusters. Other methods are based on estimated density and graph connectivity.
A special type of unsupervised learning called self-supervised learning involves training a model by generating the supervisory signal from the data itself.


==== Dimensionality reduction ====
Dimensionality reduction is a process of reducing the number of random variables under consideration by obtaining a set of principal variables. In other words, it is a process of reducing the dimension of the feature set, also called the "number of features". Most of the dimensionality reduction techniques can be considered as either feature elimination or extraction. One of the popular methods of dimensionality reduction is principal component analysis (PCA). PCA involves changing higher-dimensional data (e.g., 3D) to a smaller space (e.g., 2D).
The manifold hypothesis proposes that high-dimensional data sets lie along low-dimensional manifolds, and many dimensionality reduction techniques make this assumption, leading to the areas of manifold learning and manifold regularisation.


=== Semi-supervised learning ===

Semi-supervised learning falls between unsupervised learning (without any labelled training data) and supervised learning (with completely labelled training data). Some of the training examples are missing training labels, yet many machine-learning researchers have found that unlabelled data, when used in conjunction with a small amount of labelled data, can produce a considerable improvement in learning accuracy.
In weakly supervised learning, the training labels are noisy, limited, or imprecise; however, these labels are often cheaper to obtain, resulting in larger effective training sets.


=== Reinforcement learning ===

Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment to maximise some notion of cumulative reward. Due to its generality, the field is studied in many other disciplines, such as game theory, control theory, operations research, information theory, simulation-based optimisation, multi-agent systems, swarm intelligence, statistics and genetic algorithms. In reinforcement learning, the environment is typically represented as a Markov decision process (MDP). Many reinforcement learning algorithms use dynamic programming techniques. Reinforcement learning algorithms do not assume knowledge of an exact mathematical model of the MDP and are used when exact models are infeasible. Reinforcement learning algorithms are used in autonomous vehicles or in learning to play a game against a human opponent.


=== Other types ===
Other approaches have been developed which do not fit neatly into this three-fold categorisation, and sometimes more than one is used by the same machine learning system. For example, topic modelling, meta-learning.


==== Self-learning ====
Self-learning, as a machine learning paradigm, was introduced in 1982 along with a neural network capable of self-learning, named crossbar adaptive array (CAA). It gives a solution to the problem learning without any external reward, by introducing emotion as an internal reward. Emotion is used as a state evaluation of a self-learning agent. The CAA self-learning algorithm computes, in a crossbar fashion, both decisions about actions and emotions (feelings) about consequence situations. The system is driven by the interaction between cognition and emotion.
The self-learning algorithm updates a memory matrix W =||w(a,s)|| such that in each iteration executes the following machine learning routine: 

in situation s act a
receive a consequence situation s'
compute emotion of being in the consequence situation v(s')
update crossbar memory  w'(a,s) = w(a,s) + v(s')
It is a system with only one input, situation, and only one output, action (or behaviour) a. There is neither a separate reinforcement input nor an advice input from the environment. The backpropagated value (secondary reinforcement) is the emotion toward the consequence situation. The CAA exists in two environments, one is the behavioural environment where it behaves, and the other is the genetic environment, wherefrom it initially and only once receives initial emotions about situations to be encountered in the behavioural environment. After receiving the genome (species) vector from the genetic environment, the CAA learns a goal-seeking behaviour in an environment that contains both desirable and undesirable situations.


==== Feature learning ====

Several learning algorithms aim at discovering better representations of the inputs provided during training. Classic examples include principal component analysis and cluster analysis. Feature learning algorithms, also called representation learning algorithms, often attempt to preserve the information in their input but also transform it in a way that makes it useful, often as a pre-processing step before performing classification or predictions. This technique allows reconstruction of the inputs coming from the unknown data-generating distribution, while not being necessarily faithful to configurations that are implausible under that distribution. This replaces manual feature engineering, and allows a machine to both learn the features and use them to perform a specific task.
Feature learning can be either supervised or unsupervised. In supervised feature learning, features are learned using labelled input data. Examples include artificial neural networks, multilayer perceptrons, and supervised dictionary learning. In unsupervised feature learning, features are learned with unlabelled input data.  Examples include dictionary learning, independent component analysis, autoencoders, matrix factorisation and various forms of clustering.
Manifold learning algorithms attempt to do so under the constraint that the learned representation is low-dimensional. Sparse coding algorithms attempt to do so under the constraint that the learned representation is sparse, meaning that the mathematical model has many zeros. Multilinear subspace learning algorithms aim to learn low-dimensional representations directly from tensor representations for multidimensional data, without reshaping them into higher-dimensional vectors. Deep learning algorithms discover multiple levels of representation, or a hierarchy of features, with higher-level, more abstract features defined in terms of (or generating) lower-level features. It has been argued that an intelligent machine learns a representation that disentangles the underlying factors of variation that explain the observed data.
Feature learning is motivated by the fact that machine learning tasks such as classification often require input that is mathematically and computationally convenient to process. However, real-world data such as images, video, and sensory data have not yielded attempts to algorithmically define specific features. An alternative is to discover such features or representations through examination, without relying on explicit algorithms.


==== Sparse dictionary learning ====

Sparse dictionary learning is a feature learning method where a training example is represented as a linear combination of basis functions and assumed to be a sparse matrix. The method is strongly NP-hard and difficult to solve approximately. A popular heuristic method for sparse dictionary learning is the k-SVD algorithm. Sparse dictionary learning has been applied in several contexts. In classification, the problem is to determine the class to which a previously unseen training example belongs. For a dictionary where each class has already been built, a new training example is associated with the class that is best sparsely represented by the corresponding dictionary. Sparse dictionary learning has also been applied in image denoising. The key idea is that a clean image patch can be sparsely represented by an image dictionary, but the noise cannot.


==== Anomaly detection ====

In data mining, anomaly detection, also known as outlier detection, is the identification of rare items, events or observations that raise suspicions by differing significantly from the majority of the data. Typically, the anomalous items represent an issue such as bank fraud, a structural defect, medical problems or errors in a text. Anomalies are referred to as outliers, novelties, noise, deviations and exceptions.
In particular, in the context of abuse and network intrusion detection, the interesting objects are often not rare, but unexpected bursts of inactivity. This pattern does not adhere to the common statistical definition of an outlier as a rare object. Many outlier detection methods (in particular, unsupervised algorithms) will fail on such data unless aggregated appropriately. Instead, a cluster analysis algorithm may be able to detect the micro-clusters formed by these patterns.
Three broad categories of anomaly detection techniques exist. Unsupervised anomaly detection techniques detect anomalies in an unlabelled test data set under the assumption that the majority of the instances in the data set are normal, by looking for instances that seem to fit the least to the remainder of the data set. Supervised anomaly detection techniques require a data set that has been labelled as "normal" and "abnormal" and involves training a classifier (the key difference from many other statistical classification problems is the inherently unbalanced nature of outlier detection). Semi-supervised anomaly detection techniques construct a model representing normal behaviour from a given normal training data set and then test the likelihood of a test instance being generated by the model.


==== Robot learning ====
Robot learning is inspired by a multitude of machine learning methods, starting from supervised learning, reinforcement learning, and finally meta-learning (e.g. MAML).


==== Association rules ====

Association rule learning is a rule-based machine learning method for discovering relationships between variables in large databases. It is intended to identify strong rules discovered in databases using some measure of "interestingness".
Rule-based machine learning is a general term for any machine learning method that identifies, learns, or evolves "rules" to store, manipulate or apply knowledge. The defining characteristic of a rule-based machine learning algorithm is the identification and utilisation of a set of relational rules that collectively represent the knowledge captured by the system. This is in contrast to other machine learning algorithms that commonly identify a singular model that can be universally applied to any instance in order to make a prediction. Rule-based machine learning approaches include learning classifier systems, association rule learning, and artificial immune systems.
Based on the concept of strong rules, Rakesh Agrawal, Tomasz Imieliński and Arun Swami introduced association rules for discovering regularities between products in large-scale transaction data recorded by point-of-sale (POS) systems in supermarkets. For example, the rule 
  
    
      
        {
        
          o
          n
          i
          o
          n
          s
          ,
          p
          o
          t
          a
          t
          o
          e
          s
        
        }
        ⇒
        {
        
          b
          u
          r
          g
          e
          r
        
        }
      
    
    {\displaystyle \{\mathrm {onions,potatoes} \}\Rightarrow \{\mathrm {burger} \}}
  
 found in the sales data of a supermarket would indicate that if a customer buys onions and potatoes together, they are likely to also buy hamburger meat. Such information can be used as the basis for decisions about marketing activities such as promotional pricing or product placements. In addition to market basket analysis, association rules are employed today in application areas including Web usage mining, intrusion detection, continuous production, and bioinformatics. In contrast with sequence mining, association rule learning typically does not consider the order of items either within a transaction or across transactions.
Learning classifier systems (LCS) are a family of rule-based machine learning algorithms that combine a discovery component, typically a genetic algorithm, with a learning component, performing either supervised learning, reinforcement learning, or unsupervised learning. They seek to identify a set of context-dependent rules that collectively store and apply knowledge in a piecewise manner to make predictions.
Inductive logic programming (ILP) is an approach to rule learning using logic programming as a uniform representation for input examples, background knowledge, and hypotheses. Given an encoding of the known background knowledge and a set of examples represented as a logical database of facts, an ILP system will derive a hypothesized logic program that entails all positive and no negative examples. Inductive programming is a related field that considers any kind of programming language for representing hypotheses (and not only logic programming), such as functional programs.
Inductive logic programming is particularly useful in bioinformatics and natural language processing. Gordon Plotkin and Ehud Shapiro laid the initial theoretical foundation for inductive machine learning in a logical setting. Shapiro built their first implementation (Model Inference System) in 1981: a Prolog program that inductively inferred logic programs from positive and negative examples. The term inductive here refers to philosophical induction, suggesting a theory to explain observed facts, rather than mathematical induction, proving a property for all members of a well-ordered set.


== Models ==
A machine learning model is a type of mathematical model that, once "trained" on a given dataset, can be used to make predictions or classifications on new data. During training, a learning algorithm iteratively adjusts the model's internal parameters to minimise errors in its predictions. By extension, the term "model" can refer to several levels of specificity, from a general class of models and their associated learning algorithms to a fully trained model with all its internal parameters tuned.
Various types of models have been used and researched for machine learning systems, picking the best model for a task is called model selection.


=== Artificial neural networks ===

Artificial neural networks (ANNs), or connectionist systems, are computing systems vaguely inspired by the biological neural networks that constitute animal brains. Such systems "learn" to perform tasks by considering examples, generally without being programmed with any task-specific rules.
An ANN is a model based on a collection of connected units or nodes called "artificial neurons", which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit information, a "signal", from one artificial neuron to another. An artificial neuron that receives a signal can process it and then signal additional artificial neurons connected to it. In common ANN implementations, the signal at a connection between artificial neurons is a real number, and the output of each artificial neuron is computed by some non-linear function of the sum of its inputs. The connections between artificial neurons are called "edges". Artificial neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Artificial neurons may have a threshold such that the signal is only sent if the aggregate signal crosses that threshold. Typically, artificial neurons are aggregated into layers. Different layers may perform different kinds of transformations on their inputs. Signals travel from the first layer (the input layer) to the last layer (the output layer), possibly after traversing the layers multiple times.
The original goal of the ANN approach was to solve problems in the same way that a human brain would. However, over time, attention moved to performing specific tasks, leading to deviations from biology. Artificial neural networks have been used on a variety of tasks, including computer vision, speech recognition, machine translation, social network filtering, playing board and video games and medical diagnosis.
Deep learning consists of multiple hidden layers in an artificial neural network. This approach tries to model the way the human brain processes light and sound into vision and hearing. Some successful applications of deep learning are computer vision and speech recognition.


=== Decision trees ===

Decision tree learning uses a decision tree as a predictive model to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). It is one of the predictive modelling approaches used in statistics, data mining, and machine learning. Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels, and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. In decision analysis, a decision tree can be used to visually and explicitly represent decisions and decision making. In data mining, a decision tree describes data, but the resulting classification tree can be an input for decision-making.


=== Random forest regression ===
Random forest regression (RFR) falls under the umbrella of decision tree-based models. RFR is an ensemble learning method that builds multiple decision trees and averages their predictions to improve accuracy and to avoid overfitting. To build decision trees, RFR uses bootstrapped sampling; for instance, each decision tree is trained on random data from the training set. This random selection of RFR for training enables the model to reduce biased predictions and achieve a higher degree of accuracy. RFR generates independent decision trees, and it can work on single-output data as well as multiple regressor tasks. This makes RFR compatible to be use in various applications.


=== Support-vector machines ===

Support-vector machines (SVMs), also known as support-vector networks, are a set of related supervised learning methods used for classification and regression. Given a set of training examples, each marked as belonging to one of two categories, an SVM training algorithm builds a model that predicts whether a new example falls into one category. An SVM training algorithm is a non-probabilistic, binary, linear classifier, although methods such as Platt scaling exist to use SVM in a probabilistic classification setting. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.


=== Regression analysis ===

Regression analysis encompasses a large variety of statistical methods to estimate the relationship between input variables and their associated features. Its most common form is linear regression, where a single line is drawn to best fit the given data according to a mathematical criterion such as ordinary least squares. The latter is often extended by regularisation methods to mitigate overfitting and bias, as in ridge regression. When dealing with non-linear problems, go-to models include polynomial regression (for example, used for trendline fitting in Microsoft Excel), logistic regression (often used in statistical classification) or even kernel regression, which introduces non-linearity by taking advantage of the kernel trick to implicitly map input variables to higher-dimensional space.
Multivariate linear regression extends the concept of linear regression to handle multiple dependent variables simultaneously. This approach estimates the relationships between a set of input variables and several output variables by fitting a multidimensional linear model. It is particularly useful in scenarios where outputs are interdependent or share underlying patterns, such as predicting multiple economic indicators or reconstructing images, which are inherently multi-dimensional.


=== Bayesian networks ===

A Bayesian network, belief network, or directed acyclic graphical model is a probabilistic graphical model that represents a set of random variables and their conditional independence with a directed acyclic graph (DAG). For example, a Bayesian network could represent the probabilistic relationships between diseases and symptoms. Given symptoms, the network can be used to compute the probabilities of the presence of various diseases. Efficient algorithms exist that perform inference and learning. Bayesian networks that model sequences of variables, like speech signals or protein sequences, are called dynamic Bayesian networks. Generalisations of Bayesian networks that can represent and solve decision problems under uncertainty are called influence diagrams.


=== Gaussian processes ===

A Gaussian process is a stochastic process in which every finite collection of the random variables in the process has a multivariate normal distribution, and it relies on a pre-defined covariance function, or kernel, that models how pairs of points relate to each other depending on their locations.
Given a set of observed points, or input–output examples, the distribution of the (unobserved) output of a new point as a function of its input data can be directly computed by looking at the observed points and the covariances between those points and the new, unobserved point.
Gaussian processes are popular surrogate models in Bayesian optimisation used to do hyperparameter optimisation.


=== Genetic algorithms ===

A genetic algorithm (GA) is a search algorithm and heuristic technique that mimics the process of natural selection, using methods such as mutation and crossover to generate new genotypes in the hope of finding good solutions to a given problem. In machine learning, genetic algorithms were used in the 1980s and 1990s. Conversely, machine learning techniques have been used to improve the performance of genetic and evolutionary algorithms.


=== Belief functions ===

The theory of belief functions, also referred to as evidence theory or Dempster–Shafer theory, is a general framework for reasoning with uncertainty, with understood connections to other frameworks such as probability, possibility and  imprecise probability theories. These theoretical frameworks can be thought of as a kind of learner and have some analogous properties of how evidence is combined (e.g.,  Dempster's rule of combination), just like how in a pmf-based Bayesian approach would combine probabilities. However, there are many caveats to these beliefs functions when compared to Bayesian approaches to incorporate ignorance and uncertainty quantification. These belief function approaches that are implemented within the machine learning domain typically leverage a fusion approach of various ensemble methods to better handle the learner's decision boundary, low samples, and ambiguous class issues that standard machine learning approach tend to have difficulty resolving. However, the computational complexity of these algorithms is dependent on the number of propositions (classes), and can lead to a much higher computation time when compared to other machine learning approaches.


=== Rule-based models ===

Rule-based machine learning (RBML) is a branch of machine learning that automatically discovers and learns 'rules' from data. It provides interpretable models, making it useful for decision-making in fields like healthcare, fraud detection, and cybersecurity. Key RBML techniques includes learning classifier systems, association rule learning, artificial immune systems, and other similar models. These methods extract patterns from data and evolve rules over time.


=== Training models ===
Typically, machine learning models require a high quantity of reliable data to perform accurate predictions. When training a machine learning model, machine learning engineers need to target and collect a large and representative sample of data. Data from the training set can be as varied as a corpus of text, a collection of images, sensor data, and data collected from individual users of a service. Overfitting is something to watch out for when training a machine learning model. Trained models derived from biased or non-evaluated data can result in skewed or undesired predictions. Biased models may result in detrimental outcomes, thereby furthering the negative impacts on society or objectives. Algorithmic bias is a potential result of data not being fully prepared for training. Machine learning ethics is becoming a field of study and, notably, becoming integrated within machine learning engineering teams.


==== Federated learning ====

Federated learning is an adapted form of distributed artificial intelligence to train machine learning models that decentralises the training process, allowing for users' privacy to be maintained by not needing to send their data to a centralised server. This also increases efficiency by decentralising the training process to many devices. For example, Gboard uses federated machine learning to train search query prediction models on users' mobile phones without having to send individual searches back to Google.


== Applications ==
There are many applications for machine learning, including:

In 2006, the media-services provider Netflix held the first "Netflix Prize" competition to find a program to better predict user preferences and improve the accuracy of its existing Cinematch movie recommendation algorithm by at least 10%. A joint team made up of researchers from AT&T Labs-Research in collaboration with the teams Big Chaos and Pragmatic Theory built an ensemble model to win the Grand Prize in 2009 for $1 million. Shortly after the prize was awarded, Netflix realised that viewers' ratings were not the best indicators of their viewing patterns ("everything is a recommendation") and they changed their recommendation engine accordingly. In 2010, an article in The Wall Street Journal noted the use of machine learning by Rebellion Research to predict the 2008 financial crisis. In 2012, co-founder of Sun Microsystems, Vinod Khosla, predicted that 80% of medical doctors jobs would be lost in the next two decades to automated machine learning medical diagnostic software. In 2014, it was reported that a machine learning algorithm had been applied in the field of art history to study fine art paintings and that it may have revealed previously unrecognised influences among artists. In 2019 Springer Nature published the first research book created using machine learning. In 2020, machine learning technology was used to help make diagnoses and aid researchers in developing a cure for COVID-19. Machine learning was recently applied to predict the pro-environmental behaviour of travellers. Recently, machine learning technology was also applied to optimise smartphone's performance and thermal behaviour based on the user's interaction with the phone. When applied correctly, machine learning algorithms (MLAs) can utilise a wide range of company characteristics to predict stock returns without overfitting. By employing effective feature engineering and combining forecasts, MLAs can generate results that far surpass those obtained from basic linear techniques like OLS.
Recent advancements in machine learning have extended into the field of quantum chemistry, where novel algorithms now enable the prediction of solvent effects on chemical reactions, thereby offering new tools for chemists to tailor experimental conditions for optimal outcomes.
Machine Learning is becoming a useful tool to investigate and predict evacuation decision-making in large-scale and small-scale disasters. Different solutions have been tested to predict if and when householders decide to evacuate during wildfires and hurricanes. Other applications have been focusing on pre evacuation decisions in building fires.


== Limitations ==
Although machine learning has been transformative in some fields, machine-learning programs often fail to deliver expected results. Reasons for this are numerous: lack of (suitable) data, lack of access to the data, data bias, privacy problems, badly chosen tasks and algorithms, wrong tools and people, lack of resources, and evaluation problems.
The "black box theory" poses another yet significant challenge. Black box refers to a situation where the algorithm producing an output is entirely opaque, meaning that even the designers of an application cannot audit the pattern that the machine extracted from the data. The House of Lords Select Committee, which claimed that such an "intelligence system" that could have a "substantial impact on an individual's life" would not be considered acceptable unless it provided "a full and satisfactory explanation for the decisions" it makes.
In 2018, a self-driving car from Uber failed to detect a pedestrian, who was killed after a collision. Attempts to use machine learning in healthcare with the IBM Watson system failed to deliver even after years of time and billions of dollars invested. Microsoft's Bing Chat chatbot has been reported to produce hostile and offensive response against its users.
Machine learning has been used as a strategy to update the evidence related to a systematic review and increased reviewer burden related to the growth of biomedical literature. While it has improved with training sets, it has not yet developed sufficiently to reduce the workload burden without limiting the necessary sensitivity for the findings research itself.


=== Explainability ===

Explainable AI (XAI), or Interpretable AI, or Explainable Machine Learning (XML), is artificial intelligence (AI) in which humans can understand the decisions or predictions made by the AI. It contrasts with the "black box" concept in machine learning where even its designers cannot explain why an AI arrived at a specific decision. By refining the mental models of users of AI-powered systems and dismantling their misconceptions, XAI promises to help users perform more effectively. XAI may be an implementation of the social right to explanation.


=== Overfitting ===

Settling on a bad, overly complex theory gerrymandered to fit all the past training data is known as overfitting. Many systems attempt to reduce overfitting by rewarding a theory in accordance with how well it fits the data but penalising the theory in accordance with how complex the theory is.


=== Model collapse ===


=== Hallucinations ===


=== Other limitations and vulnerabilities ===
Learners can also be disappointed by "learning the wrong lesson". A toy example is that an image classifier trained only on pictures of brown horses and black cats might conclude that all brown patches are likely to be horses. A real-world example is that, unlike humans, current image classifiers often do not primarily make judgments from the spatial relationship between components of the picture, and they learn relationships between pixels that humans are oblivious to, but that still correlate with images of certain types of real objects. Modifying these patterns on a legitimate image can result in "adversarial" images that the system misclassifies.
Adversarial vulnerabilities can also result in nonlinear systems or from non-pattern perturbations. For some systems, it is possible to change the output by only changing a single adversarially chosen pixel. Machine learning models are often vulnerable to manipulation or evasion via adversarial machine learning.
Researchers have demonstrated how backdoors can be placed undetectably into classifying (e.g., for categories "spam" and "not spam" of posts) machine learning models that are often developed or trained by third parties. Parties can change the classification of any input, including in cases for which a type of data/software transparency is provided, possibly including white-box access.


== Model assessments ==
Classification of machine learning models can be validated by accuracy estimation techniques like the holdout method, which splits the data into a training and test set (conventionally 2/3 training set and 1/3 test set designation) and evaluates the performance of the training model on the test set. In comparison, the K-fold-cross-validation method randomly partitions the data into K subsets and then K experiments are performed each considering 1 subset for evaluation and the remaining K-1 subsets for training the model. In addition to the holdout and cross-validation methods, bootstrap, which samples n instances with replacement from the dataset, can be used to assess model accuracy.
In addition to overall accuracy, investigators frequently report sensitivity and specificity, meaning true positive rate (TPR) and true negative rate (TNR), respectively. Similarly, investigators sometimes report the false positive rate (FPR) as well as the false negative rate (FNR). However, these rates are ratios that fail to reveal their numerators and denominators. Receiver operating characteristic (ROC), along with the accompanying Area Under the ROC Curve (AUC), offer additional tools for classification model assessment. Higher AUC is associated with a better performing model.


== Ethics ==


=== Bias ===

Different machine learning approaches can suffer from different data biases. A machine learning system trained specifically on current customers may not be able to predict the needs of new customer groups that are not represented in the training data. When trained on human-made data, machine learning is likely to pick up the constitutional and unconscious biases already present in society.
Systems that are trained on datasets collected with biases may exhibit these biases upon use (algorithmic bias), thus digitising cultural prejudices. For example, in 1988, the UK's Commission for Racial Equality found that St. George's Medical School had been using a computer program trained from data of previous admissions staff and this program had denied nearly 60 candidates who were found to either be women or have non-European-sounding names. Using job hiring data from a firm with racist hiring policies may lead to a machine learning system duplicating the bias by scoring job applicants by similarity to previous successful applicants. Another example includes predictive policing company Geolitica's predictive algorithm that resulted in "disproportionately high levels of over-policing in low-income and minority communities" after being trained with historical crime data.
While responsible collection of data and documentation of algorithmic rules used by a system is considered a critical part of machine learning, some researchers blame the lack of participation and representation of minority populations in the field of AI for machine learning's vulnerability to biases. In fact, according to research carried out by the Computing Research Association in 2021, "female faculty make up just 16.1%" of all faculty members who focus on AI among several universities around the world. Furthermore, among the group of "new U.S. resident AI PhD graduates," 45% identified as white, 22.4% as Asian, 3.2% as Hispanic, and 2.4% as African American, which further demonstrates a lack of diversity in the field of AI.
Language models learned from data have been shown to contain human-like biases. Because human languages contain biases, machines trained on language corpora will necessarily also learn these biases. In 2016, Microsoft tested Tay, a chatbot that learned from Twitter, and it quickly picked up racist and sexist language.
In an experiment carried out by ProPublica, a machine learning algorithm's insight into the recidivism rates among prisoners falsely flagged "black defendants high risk twice as often as white defendants". In 2015, Google Photos once tagged a couple of black people as gorillas, which caused controversy. The gorilla label was subsequently removed, and in 2023, it still cannot recognise gorillas. Similar issues with recognising non-white people have been found in many other systems.


=== Financial incentives ===
There are concerns among health care professionals that these systems might not be designed in the public's interest but as income-generating machines. This is especially true in the United States, where there is a long-standing ethical dilemma of improving health care, but also increasing profits. For example, the algorithms could be designed to provide patients with unnecessary tests or medication in which the algorithm's proprietary owners hold stakes. There is potential for machine learning in health care to provide professionals with an additional tool to diagnose, medicate, and plan recovery paths for patients, but this requires these biases to be mitigated.


== Hardware ==
Since the 2010s, advances in both machine learning algorithms and computer hardware have led to more efficient methods for training deep neural networks (a particular narrow subdomain of machine learning) that contain many layers of nonlinear hidden units. By 2019, graphics processing units (GPUs), often with AI-specific enhancements, had displaced CPUs as the dominant method of training large-scale commercial cloud AI. OpenAI estimated the hardware compute used in the largest deep learning projects from AlexNet (2012) to AlphaZero (2017), and found a 300,000-fold increase in the amount of compute required, with a doubling-time trendline of 3.4 months.


=== Tensor Processing Units (TPUs) ===
Tensor Processing Units (TPUs) are specialised hardware accelerators developed by Google specifically for machine learning workloads. Unlike general-purpose GPUs and FPGAs, TPUs are optimised for tensor computations, making them particularly efficient for deep learning tasks such as training and inference. They are widely used in Google Cloud AI services and large-scale machine learning models like Google's DeepMind AlphaFold and large language models. TPUs leverage matrix multiplication units and high-bandwidth memory to accelerate computations while maintaining energy efficiency. Since their introduction in 2016, TPUs have become a key component of AI infrastructure, especially in cloud-based environments.


=== Neuromorphic computing ===
Neuromorphic computing refers to a class of computing systems designed to emulate the structure and functionality of biological neural networks. These systems may be implemented through software-based simulations on conventional hardware or through specialised hardware architectures.


==== Physical neural networks ====
A physical neural network is a specific type of neuromorphic hardware that relies on electrically adjustable materials, such as memristors, to emulate the function of neural synapses. The term "physical neural network" highlights the use of physical hardware for computation, as opposed to software-based implementations. It broadly refers to artificial neural networks that use materials with adjustable resistance to replicate neural synapses.


=== Embedded machine learning ===
Embedded machine learning is a sub-field of machine learning where models are deployed on embedded systems with limited computing resources, such as wearable computers, edge devices and microcontrollers. Running models directly on these devices eliminates the need to transfer and store data on cloud servers for further processing, thereby reducing the risk of data breaches, privacy leaks and theft of intellectual property, personal data and business secrets. Embedded machine learning can be achieved through various techniques, such as hardware acceleration, approximate computing, and model optimization. Common optimization techniques include pruning, quantisation, knowledge distillation, low-rank factorisation, network architecture search, and parameter sharing.


== Software ==


=== Free and open-source software ===


=== Proprietary software with free and open-source editions ===
KNIME
RapidMiner


=== Proprietary software ===


== Journals ==
Journal of Machine Learning Research
Machine Learning
Nature Machine Intelligence
Neural Computation
IEEE Transactions on Pattern Analysis and Machine Intelligence


== Conferences ==
AAAI Conference on Artificial Intelligence
Association for Computational Linguistics (ACL)
European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)
International Conference on Computational Intelligence Methods for Bioinformatics and Biostatistics (CIBB)
International Conference on Machine Learning (ICML)
International Conference on Learning Representations (ICLR)
International Conference on Intelligent Robots and Systems (IROS)
Conference on Knowledge Discovery and Data Mining (KDD)
Conference on Neural Information Processing Systems (NeurIPS)


== See also ==
Automated machine learning – Process of automating the application of machine learning
Big data – Extremely large or complex datasets
Deep learning — branch of ML concerned with artificial neural networks
Differentiable programming – Programming paradigm
Google Colab – no setup online IDE for machine learning in Python and Julia
List of datasets for machine-learning research
List of machine learning algorithms and List of algorithms for machine learning and statistical classification
M-theory (learning framework) – Framework in machine learning
Machine unlearning – Field of study in artificial intelligence
Outline of machine learning
Solomonoff's theory of inductive inference – Mathematical theory
TinyML — Machine learning on low-power embedded systems


== References ==


== Sources ==
Domingos, Pedro (22 September 2015). The Master Algorithm: How the Quest for the Ultimate Learning Machine Will Remake Our World. Basic Books. ISBN 978-0-465-06570-7.
Nilsson, Nils (1998). Artificial Intelligence: A New Synthesis. Morgan Kaufmann. ISBN 978-1-55860-467-4. Archived from the original on 26 July 2020. Retrieved 18 November 2019.
Poole, David; Mackworth, Alan; Goebel, Randy (1998). Computational Intelligence: A Logical Approach. New York: Oxford University Press. ISBN 978-0-19-510270-3. Archived from the original on 26 July 2020. Retrieved 22 August 2020.
Russell, Stuart J.; Norvig, Peter (2003), Artificial Intelligence: A Modern Approach (2nd ed.), Upper Saddle River, New Jersey: Prentice Hall, ISBN 0-13-790395-2.


== Further reading ==


== External links ==
International Machine Learning Society
MLOSS page at JMLR - an academic database of open-source machine learning software.

Physics is the scientific study of matter, its fundamental constituents, its motion and behavior through space and time, and the related entities of energy and force. It is one of the most fundamental scientific disciplines. A scientist who specializes in the field of physics is called a physicist.
Physics is one of the oldest academic disciplines. Over much of the past two millennia, physics, chemistry, biology, and certain branches of mathematics were part of natural philosophy, but during the Scientific Revolution in the 17th century, these natural sciences branched into separate research endeavors. Physics intersects with many interdisciplinary areas of research, such as biophysics and quantum chemistry, and the boundaries of physics are not rigidly defined. New ideas in physics often explain the fundamental mechanisms studied by other sciences and suggest new avenues of research in these and other academic disciplines, such as mathematics and philosophy.
Advances in physics often enable new technologies. For example, advances in the understanding of electromagnetism, solid-state physics, and nuclear physics led directly to the development of technologies that have transformed modern society, such as television, computers, domestic appliances, and nuclear weapons; advances in thermodynamics led to the development of industrialization; and advances in mechanics inspired the development of calculus.


== History ==

The word physics comes from the Latin physica ('study of nature'), which itself is a borrowing of the Greek φυσική (phusikḗ 'natural science'), a term derived from φύσις (phúsis 'origin, nature, property').


=== Ancient astronomy ===

Astronomy is one of the oldest natural sciences. Early civilizations dating before 3000 BCE, such as the Sumerians, ancient Egyptians, and the Indus Valley Civilization, had a predictive knowledge and a basic awareness of the motions of the Sun, Moon, and stars. The stars and planets, believed to represent gods, were often worshipped. While the explanations for the observed positions of the stars were often unscientific and lacking in evidence, these early observations laid the foundation for later astronomy, as the stars were found to traverse great circles across the sky, which could not explain the positions of the planets.
According to Asger Aaboe, the origins of Western astronomy can be found in Mesopotamia, and all Western efforts in the exact sciences are descended from late Babylonian astronomy. Egyptian astronomers left monuments showing knowledge of the constellations and the motions of the celestial bodies, while Greek poet Homer wrote of various celestial objects in his Iliad and Odyssey; later Greek astronomers provided names, which are still used today, for most constellations visible from the Northern Hemisphere.


=== Natural philosophy ===

Natural philosophy has its origins in Greece during the Archaic period (650 BCE – 480 BCE), when pre-Socratic philosophers like Thales rejected non-naturalistic explanations for natural phenomena and proclaimed that every event had a natural cause. They proposed ideas verified by reason and observation, and many of their hypotheses proved successful in experiment; for example, atomism was found to be correct approximately 2000 years after it was proposed by Leucippus and his pupil Democritus.


=== Aristotle and Hellenistic physics ===

During the classical period in Greece (6th, 5th and 4th centuries BCE) and in Hellenistic times, natural philosophy developed along many lines of inquiry. Aristotle (Greek: Ἀριστοτέλης, Aristotélēs) (384–322 BCE), a student of Plato,
wrote on many subjects, including a substantial treatise on "Physics" – in the 4th century BC. Aristotelian physics was influential for about two millennia. His approach mixed some limited observation with logical deductive arguments, but did not rely on experimental verification of deduced statements. Aristotle's foundational work in Physics, though very imperfect, formed a framework against which later thinkers further developed the field. His approach is entirely superseded today.
He explained ideas such as motion (and gravity) with the theory of four elements.
Aristotle believed that each of the four classical elements (air, fire, water, earth) had its own natural place. Because of their differing densities, each element will revert to its own specific place in the atmosphere. So, because of their weights, fire would be at the top, air underneath fire, then water, then lastly earth. He also stated that when a small amount of one element enters the natural place of another, the less abundant element will automatically go towards its own natural place. For example, if there is a fire on the ground, the flames go up into the air in an attempt to go back into its natural place where it belongs. His laws of motion included: that heavier objects will fall faster, the speed being proportional to the weight and the speed of the object that is falling depends inversely on the density object it is falling through (e.g. density of air). He also stated that, when it comes to violent motion (motion of an object when a force is applied to it by a second object), the speed that object moves will only be as fast or strong as the measure of force applied to it. The problem of motion and its causes was studied carefully, leading to the philosophical notion of a "prime mover" as the ultimate source of all motion in the world (Book 8 of his treatise Physics).


=== Medieval European and Islamic ===

The Western Roman Empire fell to invaders and internal decay in the fifth century, resulting in a decline in intellectual pursuits in western Europe. By contrast, the Eastern Roman Empire (usually known as the Byzantine Empire) resisted the attacks from invaders and continued to advance various fields of learning, including physics. In the sixth century, John Philoponus challenged the dominant Aristotelian approach to science although much of his work was focused on Christian theology.
In the sixth century, Isidore of Miletus created an important compilation of Archimedes' works that are copied in the Archimedes Palimpsest.
Islamic scholarship inherited Aristotelian physics from the Greeks and during the Islamic Golden Age developed it further.
The most notable innovations under Islamic scholarship were in the field of optics and vision, which came from the works of many scientists like Ibn Sahl, Al-Kindi, Ibn al-Haytham, Al-Farisi and Avicenna. In his Book of Optics (also known as Kitāb al-Manāẓir) Ibn al-Haytham presented the idea of light rays as an alternative to the ancient Greek idea about visual rays. Like Ptolemy, Ibn al-Haytham applied controlled experiments, verifying the laws of refraction and reflection with the new concept of light rays, but still lacking the concept of image formation.


=== Scientific Revolution ===

Physics became a separate science when early modern Europeans used experimental and quantitative methods to discover what are now considered to be the laws of physics.
Major developments in this period include the replacement of the geocentric model of the Solar System with the heliocentric Copernican model, the laws governing the motion of planetary bodies (determined by Johannes Kepler between 1609 and 1619), Galileo's pioneering work on telescopes and observational astronomy in the 16th and 17th centuries, and Isaac Newton's discovery and unification of the laws of motion and universal gravitation (that would come to bear his name). Newton, and separately Gottfried Wilhelm Leibniz, developed calculus, the mathematical study of continuous change, and Newton applied it to solve physical problems.


=== 19th century ===

The discovery of laws in thermodynamics, chemistry, and electromagnetics resulted from research efforts during the Industrial Revolution as energy needs increased. By the end of the 19th century, theories of thermodynamics, mechanics, and electromagnetics matched a wide variety of observations. Taken together these theories became the basis for what would later be called classical physics.
A few experimental results remained inexplicable. Classical electromagnetism presumed a medium, an luminiferous aether to support the propagation of waves, but this medium could not be detected. The intensity of light from hot glowing blackbody objects did not match the predictions of thermodynamics and electromagnetism. The character of electron emission of illuminated metals differed from predictions. These failures, seemingly insignificant in the big picture would upset the physics world in first two decades of the 20th century.


=== 20th century ===

Modern physics began in the early 20th century with the work of Max Planck in quantum theory and Albert Einstein's theory of relativity. Both of these theories came about due to inaccuracies in classical mechanics in certain situations. Classical mechanics predicted that the speed of light depends on the motion of the observer, which could not be resolved with the constant speed predicted by Maxwell's equations of electromagnetism. This discrepancy was corrected by Einstein's theory of special relativity, which replaced classical mechanics for fast-moving bodies and allowed for a constant speed of light. Black-body radiation provided another problem for classical physics, which was corrected when Planck proposed that the excitation of material oscillators is possible only in discrete steps proportional to their frequency. This, along with the photoelectric effect and a complete theory predicting discrete energy levels of electron orbitals, led to the theory of quantum mechanics improving on classical physics at very small scales.
Quantum mechanics would come to be pioneered by Werner Heisenberg, Erwin Schrödinger and Paul Dirac. From this early work, and work in related fields, the Standard Model of particle physics was derived. Following the discovery of a particle with properties consistent with the Higgs boson at CERN in 2012, all fundamental particles predicted by the Standard Model, and no others, appear to exist; however, physics beyond the Standard Model, with theories such as supersymmetry, is an active area of research. Areas of mathematics in general are important to this field, such as the study of probabilities and groups.


== Core theories ==

Physics deals with a wide variety of systems, although certain theories are used by all physicists. Each of these theories was experimentally tested numerous times and found to be an adequate approximation of nature.
These central theories are important tools for research into more specialized topics, and any physicist, regardless of their specialization, is expected to be literate in them. These include classical mechanics, quantum mechanics, thermodynamics and statistical mechanics, electromagnetism, and special relativity.


=== Distinction between classical and modern physics ===

In the first decades of the 20th century physics was revolutionized by the discoveries of quantum mechanics and relativity. The changes were so fundamental that these new concepts became the foundation of "modern physics", with other topics becoming "classical physics". The majority of applications of physics are essentially classical. 
The laws of classical physics accurately describe systems whose important length scales are greater than the atomic scale and whose motions are much slower than the speed of light. Outside of this domain, observations do not match predictions provided by classical mechanics.


=== Classical theory ===

Classical physics includes the traditional branches and topics that were recognized and well-developed before the beginning of the 20th century—classical mechanics, thermodynamics, and electromagnetism. Classical mechanics is concerned with bodies acted on by forces and bodies in motion and may be divided into statics (study of the forces on a body or bodies not subject to an acceleration), kinematics (study of motion without regard to its causes), and dynamics (study of motion and the forces that affect it); mechanics may also be divided into solid mechanics and fluid mechanics (known together as continuum mechanics), the latter include such branches as hydrostatics, hydrodynamics and pneumatics. Acoustics is the study of how sound is produced, controlled, transmitted and received. Important modern branches of acoustics include ultrasonics, the study of sound waves of very high frequency beyond the range of human hearing; bioacoustics, the physics of animal calls and hearing; and electroacoustics, the manipulation of audible sound waves using electronics.
Optics, the study of light, is concerned not only with visible light but also with infrared and ultraviolet radiation, which exhibit all of the phenomena of visible light except visibility, e.g., reflection, refraction, interference, diffraction, dispersion, and polarization of light. Heat is a form of energy, the internal energy possessed by the particles of which a substance is composed; thermodynamics deals with the relationships between heat and other forms of energy. Electricity and magnetism have been studied as a single branch of physics since the intimate connection between them was discovered in the early 19th century; an electric current gives rise to a magnetic field, and a changing magnetic field induces an electric current. Electrostatics deals with electric charges at rest, electrodynamics with moving charges, and magnetostatics with magnetic poles at rest.


=== Modern theory ===

The discovery of relativity and of quantum mechanics in the first decades of the 20th century transformed the conceptual basis of physics without reducing the practical value of most of the physical theories developed up to that time. Consequently the topics of physics have come to be divided into "classical physics" and "modern physics", with the latter category including effects related to quantum mechanics and relativity.
Classical physics is generally concerned with matter and energy on the normal scale of observation, while much of modern physics is concerned with the behavior of matter and energy under extreme conditions or on a very large or very small scale. For example, atomic and nuclear physics study matter on the smallest scale at which chemical elements can be identified. The physics of elementary particles is on an even smaller scale since it is concerned with the most basic units of matter; this branch of physics is also known as high-energy physics because of the extremely high energies necessary to produce many types of particles in particle accelerators. On this scale, ordinary, commonsensical notions of space, time, matter, and energy are no longer valid.
The two chief theories of modern physics present a different picture of the concepts of space, time, and matter from that presented by classical physics. Classical mechanics approximates nature as continuous, while quantum theory is concerned with the discrete nature of many phenomena at the atomic and subatomic level and with the complementary aspects of particles and waves in the description of such phenomena. The theory of relativity is concerned with the description of phenomena that take place in a frame of reference that is in motion with respect to an observer; the special theory of relativity is concerned with motion in the absence of gravitational fields and the general theory of relativity with motion and its connection with gravitation. Both quantum theory and the theory of relativity find applications in many areas of modern physics.
Fundamental concepts in modern physics include:

Action
Causality
Covariance
Particle
Physical field
Physical interaction
Quantum
Statistical ensemble
Symmetry
Wave


== Research ==


=== Scientific method ===
Physicists use the scientific method to test the validity of a physical theory. By using a methodical approach to compare the implications of a theory with the conclusions drawn from its related experiments and observations, physicists are better able to test the validity of a theory in a logical, unbiased, and repeatable way. To that end, experiments are performed and observations are made in order to determine the validity or invalidity of a theory.
A scientific law is a concise verbal or mathematical statement of a relation that expresses a fundamental principle of some theory, such as Newton's law of universal gravitation.


=== Theory and experiment ===

Theorists seek to develop mathematical models that both agree with existing experiments and successfully predict future experimental results, while experimentalists devise and perform experiments to test theoretical predictions and explore new phenomena. Although theory and experiment are developed separately, they strongly affect and depend upon each other. Progress in physics frequently comes about when experimental results defy explanation by existing theories, prompting intense focus on applicable modeling, and when new theories generate experimentally testable predictions, which inspire the development of new experiments (and often related equipment).
Physicists who work at the interplay of theory and experiment are called phenomenologists, who study complex phenomena observed in experiment and work to relate them to a fundamental theory.
Theoretical physics has historically taken inspiration from philosophy; electromagnetism was unified this way. Beyond the known universe, the field of theoretical physics also deals with hypothetical issues, such as parallel universes, a multiverse, and higher dimensions. Theorists invoke these ideas in hopes of solving particular problems with existing theories; they then explore the consequences of these ideas and work toward making testable predictions.
Experimental physics expands, and is expanded by, engineering and technology. Experimental physicists who are involved in basic research design and perform experiments with equipment such as particle accelerators and lasers, whereas those involved in applied research often work in industry, developing technologies such as magnetic resonance imaging (MRI) and transistors. Feynman has noted that experimentalists may seek areas that have not been explored well by theorists.


=== Scope and aims ===

Physics covers a wide range of phenomena, from elementary particles (such as quarks, neutrinos, and electrons) to the largest superclusters of galaxies. Included in these phenomena are the most basic objects composing all other things. Therefore, physics is sometimes called the "fundamental science". Physics aims to describe the various phenomena that occur in nature in terms of simpler phenomena. Thus, physics aims to both connect the things observable to humans to root causes, and then connect these causes together.
For example, the ancient Chinese observed that certain rocks (lodestone and magnetite) were attracted to one another by an invisible force. This effect was later called magnetism, which was first rigorously studied in the 17th century. But even before the Chinese discovered magnetism, the ancient Greeks knew of other objects such as amber, that when rubbed with fur would cause a similar invisible attraction between the two. This was also first studied rigorously in the 17th century and came to be called electricity. Thus, physics had come to understand two observations of nature in terms of some root cause (electricity and magnetism). However, further work in the 19th century revealed that these two forces were just two different aspects of one force—electromagnetism. This process of "unifying" forces continues today, and electromagnetism and the weak nuclear force are now considered to be two aspects of the electroweak interaction. Physics hopes to find an ultimate reason (theory of everything) for why nature is as it is (see section Current research below for more information).


=== Current research ===

Research in physics is continually progressing on a large number of fronts.
In condensed matter physics, an important unsolved theoretical problem is that of high-temperature superconductivity. Many condensed matter experiments are aiming to fabricate workable spintronics and quantum computers.
In particle physics, the first pieces of experimental evidence for physics beyond the Standard Model have begun to appear. Foremost among these are indications that neutrinos have non-zero mass. These experimental results appear to have solved the long-standing solar neutrino problem, and the physics of massive neutrinos remains an area of active theoretical and experimental research. The Large Hadron Collider has already found the Higgs boson, but future research aims to prove or disprove the supersymmetry, which extends the Standard Model of particle physics. Research on the nature of the major mysteries of dark matter and dark energy is also currently ongoing.
Although much progress has been made in high-energy, quantum, and astronomical physics, many everyday phenomena involving complexity, chaos, or turbulence are still poorly understood. Complex problems that seem like they could be solved by a clever application of dynamics and mechanics remain unsolved; examples include the formation of sandpiles, nodes in trickling water, the shape of water droplets, mechanisms of surface tension catastrophes, and self-sorting in shaken heterogeneous collections.
These complex phenomena have received growing attention since the 1970s for several reasons, including the availability of modern mathematical methods and computers, which enabled complex systems to be modeled in new ways. Complex physics has become part of increasingly interdisciplinary research, as exemplified by the study of turbulence in aerodynamics and the observation of pattern formation in biological systems. In the 1932 Annual Review of Fluid Mechanics, Horace Lamb said:

I am an old man now, and when I die and go to heaven there are two matters on which I hope for enlightenment. One is quantum electrodynamics, and the other is the turbulent motion of fluids. And about the former I am rather optimistic.


== Branches and fields ==


=== Fields ===
The major fields of physics, along with their subfields and the theories and concepts they employ, are shown in the following table.

Since the 20th century, the individual fields of physics have become increasingly specialized, and today most physicists work in a single field for their entire careers. "Universalists" such as Einstein (1879–1955) and Lev Landau (1908–1968), who worked in multiple fields of physics, are now very rare.
Contemporary research in physics can be broadly divided into nuclear and particle physics; condensed matter physics; atomic, molecular, and optical physics; astrophysics; and applied physics. Some physics departments also support physics education research and physics outreach.


==== Nuclear and particle ====

Particle physics is the study of the elementary constituents of matter and energy and the interactions between them. In addition, particle physicists design and develop the high-energy accelerators, detectors, and computer programs necessary for this research. The field is also called "high-energy physics" because many elementary particles do not occur naturally but are created only during high-energy collisions of other particles.
Currently, the interactions of elementary particles and fields are described by the Standard Model. The model accounts for the 12 known particles of matter (quarks and leptons) that interact via the strong, weak, and electromagnetic fundamental forces. Dynamics are described in terms of matter particles exchanging gauge bosons (gluons, W and Z bosons, and photons, respectively). The Standard Model also predicts a particle known as the Higgs boson. In July 2012 CERN, the European laboratory for particle physics, announced the detection of a particle consistent with the Higgs boson, an integral part of the Higgs mechanism.
Nuclear physics is the field of physics that studies the constituents and interactions of atomic nuclei. The most commonly known applications of nuclear physics are nuclear power generation and nuclear weapons technology, but the research has provided application in many fields, including those in nuclear medicine and magnetic resonance imaging, ion implantation in materials engineering, and radiocarbon dating in geology and archaeology.


==== Atomic, molecular, and optical ====

Atomic, molecular, and optical physics (AMO) is the study of matter—matter and light—matter interactions on the scale of single atoms and molecules. The three areas are grouped together because of their interrelationships, the similarity of methods used, and the commonality of their relevant energy scales. All three areas include both classical, semi-classical and quantum treatments; they can treat their subject from a microscopic view (in contrast to a macroscopic view).
Atomic physics studies the electron shells of atoms. Current research focuses on activities in quantum control, cooling and trapping of atoms and ions, low-temperature collision dynamics and the effects of electron correlation on structure and dynamics. Atomic physics is influenced by the nucleus (see hyperfine splitting), but intra-nuclear phenomena such as fission and fusion are considered part of nuclear physics.
Molecular physics focuses on multi-atomic structures and their internal and external interactions with matter and light. Optical physics is distinct from optics in that it tends to focus not on the control of classical light fields by macroscopic objects but on the fundamental properties of optical fields and their interactions with matter in the microscopic realm.


==== Condensed matter ====

Condensed matter physics is the field of physics that deals with the macroscopic physical properties of matter. In particular, it is concerned with the "condensed" phases that appear whenever the number of particles in a system is extremely large and the interactions between them are strong.
The most familiar examples of condensed phases are solids and liquids, which arise from the bonding by way of the electromagnetic force between atoms. More exotic condensed phases include the superfluid and the Bose–Einstein condensate found in certain atomic systems at very low temperature, the superconducting phase exhibited by conduction electrons in certain materials, and the ferromagnetic and antiferromagnetic phases of spins on atomic lattices.
Condensed matter physics is the largest field of contemporary physics. Historically, condensed matter physics grew out of solid-state physics, which is now considered one of its main subfields. The term condensed matter physics was apparently coined by Philip Anderson when he renamed his research group—previously solid-state theory—in 1967. In 1978, the Division of Solid State Physics of the American Physical Society was renamed as the Division of Condensed Matter Physics. Condensed matter physics has a large overlap with chemistry, materials science, nanotechnology and engineering.


==== Astrophysics ====

Astrophysics and astronomy are the application of the theories and methods of physics to the study of stellar structure, stellar evolution, the origin of the Solar System, and related problems of cosmology. Because astrophysics is a broad subject, astrophysicists typically apply many disciplines of physics, including mechanics, electromagnetism, statistical mechanics, thermodynamics, quantum mechanics, relativity, nuclear and particle physics, and atomic and molecular physics.
The discovery by Karl Jansky in 1931 that radio signals were emitted by celestial bodies initiated the science of radio astronomy. Most recently, the frontiers of astronomy have been expanded by space exploration. Perturbations and interference from the Earth's atmosphere make space-based observations necessary for infrared, ultraviolet, gamma-ray, and X-ray astronomy.
Physical cosmology is the study of the formation and evolution of the universe on its largest scales. Albert Einstein's theory of relativity plays a central role in all modern cosmological theories. In the early 20th century, Hubble's discovery that the universe is expanding, as shown by the Hubble diagram, prompted rival explanations known as the steady state universe and the Big Bang.
The Big Bang was confirmed by the success of Big Bang nucleosynthesis and the discovery of the cosmic microwave background in 1964. The Big Bang model rests on two theoretical pillars: Albert Einstein's general relativity and the cosmological principle. Cosmologists have recently established the ΛCDM model of the evolution of the universe, which includes cosmic inflation, dark energy, and dark matter.


== Other aspects ==


=== Education ===


=== Careers ===


=== Philosophy ===

Physics, as with the rest of science, relies on the philosophy of science and its "scientific method" to advance knowledge of the physical world. The scientific method employs a priori and a posteriori reasoning as well as the use of Bayesian inference to measure the validity of a given theory.
Study of the philosophical issues surrounding physics, the philosophy of physics, involves issues such as the nature of space and time, determinism, and metaphysical outlooks such as empiricism, naturalism, and realism.
Many physicists have written about the philosophical implications of their work, for instance Laplace, who championed causal determinism, and Erwin Schrödinger, who wrote on quantum mechanics. The mathematical physicist Roger Penrose has been called a Platonist by Stephen Hawking, a view Penrose discusses in his book, The Road to Reality. Hawking referred to himself as an "unashamed reductionist" and took issue with Penrose's views.

Mathematics provides a compact and exact language used to describe the order in nature. This was noted and advocated by Pythagoras, Plato, Galileo, and Newton. Some theorists, like Hilary Putnam and Penelope Maddy, hold that logical truths, and therefore mathematical reasoning, depend on the empirical world. This is usually combined with the claim that the laws of logic express universal regularities found in the structural features of the world, which may explain the peculiar relation between these fields.
Physics uses mathematics to organize and formulate experimental results. From those results, precise or estimated solutions are obtained, or quantitative results, from which new predictions can be made and experimentally confirmed or negated. The results from physics experiments are numerical data, with their units of measure and estimates of the errors in the measurements. Technologies based on mathematics, like computation have made computational physics an active area of research.

Ontology is a prerequisite for physics, but not for mathematics. It means physics is ultimately concerned with descriptions of the real world, while mathematics is concerned with abstract patterns, even beyond the real world. Thus physics statements are synthetic, while mathematical statements are analytic. Mathematics contains hypotheses, while physics contains theories. Mathematics statements have to be only logically true, while predictions of physics statements must match observed and experimental data.
The distinction is clear-cut, but not always obvious. For example, mathematical physics is the application of mathematics in physics. Its methods are mathematical, but its subject is physical. The problems in this field start with a "mathematical model of a physical situation" (system) and a "mathematical description of a physical law" that will be applied to that system. Every mathematical statement used for solving has a hard-to-find physical meaning. The final mathematical solution has an easier-to-find meaning, because it is what the solver is looking for.


=== Fundamental vs. applied physics ===

Physics is a branch of fundamental science (also called basic science). Physics is also called "the fundamental science" because all branches of natural science including chemistry, astronomy, geology, and biology are constrained by laws of physics. Similarly, chemistry is often called the central science because of its role in linking the physical sciences. For example, chemistry studies properties, structures, and reactions of matter (chemistry's focus on the molecular and atomic scale distinguishes it from physics). Structures are formed because particles exert electrical forces on each other, properties include physical characteristics of given substances, and reactions are bound by laws of physics, like conservation of energy, mass, and charge. Fundamental physics seeks to better explain and understand phenomena in all spheres, without a specific practical application as a goal, other than the deeper insight into the phenomema themselves.

Applied physics is a general term for physics research and development that is intended for a particular use. An applied physics curriculum usually contains a few classes in an applied discipline, like geology or electrical engineering. It usually differs from engineering in that an applied physicist may not be designing something in particular, but rather is using physics or conducting physics research with the aim of developing new technologies or solving a problem.
The approach is similar to that of applied mathematics. Applied physicists use physics in scientific research. For instance, people working on accelerator physics might seek to build better particle detectors for research in theoretical physics.
Physics is used heavily in engineering. For example, statics, a subfield of mechanics, is used in the building of bridges and other static structures. The understanding and use of acoustics results in sound control and better concert halls; similarly, the use of optics creates better optical devices. An understanding of physics makes for more realistic flight simulators, video games, and movies, and is often critical in forensic investigations.

With the standard consensus that the laws of physics are universal and do not change with time, physics can be used to study things that would ordinarily be mired in uncertainty. For example, in the study of the origin of the Earth, a physicist can reasonably model Earth's mass, temperature, and rate of rotation, as a function of time allowing the extrapolation forward or backward in time and so predict future or prior events. It also allows for simulations in engineering that speed up the development of a new technology.
There is also considerable interdisciplinarity, so many other important fields are influenced by physics (e.g., the fields of econophysics and sociophysics).


== See also ==

Earth science – Fields of natural science related to Earth
Neurophysics – Study of the nervous system with physics
Psychophysics – Branch of knowledge relating physical stimuli and psychological perception
Relationship between mathematics and physics – Relationship between fields of study
Science tourism – Travel to notable science locations


=== Lists ===
List of important publications in physics
List of physicists
Lists of physics equations


== Notes ==


== References ==


== Sources ==


== External links ==

Physics at Quanta Magazine
Usenet Physics FAQ – FAQ compiled by sci.physics and other physics newsgroups
Website of the Nobel Prize in physics Archived 7 December 2021 at the Wayback Machine – Award for outstanding contributions to the subject
World of Physics Archived 25 June 2025 at the Wayback Machine – Online encyclopedic dictionary of physics
Nature Physics – Academic journal
Physics Archived 28 June 2025 at the Wayback Machine – Online magazine by the American Physical Society
The Vega Science Trust Archived 7 June 2023 at the Wayback Machine – Science videos, including physics
HyperPhysics website Archived 8 April 2011 at the Wayback Machine – Physics and astronomy mind-map from Georgia State University
Physics at MIT OpenCourseWare Archived 15 March 2022 at the Wayback Machine – Online course material from Massachusetts Institute of Technology
The Feynman Lectures on Physics Archived 4 March 2022 at the Wayback Machine

In mathematics and computer science, an algorithm ( ) is a finite sequence of mathematically rigorous instructions, typically used to solve a class of specific problems or to perform a computation. Algorithms are used as specifications for performing calculations and data processing. More advanced algorithms can use conditionals to divert the code execution through various routes (referred to as automated decision-making) and deduce valid inferences (referred to as automated reasoning).
In contrast, a heuristic is an approach to solving problems without well-defined correct or optimal results. For example, although social media recommender systems are commonly called "algorithms", they actually rely on heuristics as there is no truly "correct" recommendation.
As an effective method, an algorithm can be expressed within a finite amount of space and time and in a well-defined formal language for calculating a function. Starting from an initial state and input, a computation occurs at each step, eventually producing output and terminating. The transition between states can be non-deterministic; randomized algorithms incorporate random input.


== Etymology ==
Around 825 AD, Persian scientist and polymath Muḥammad ibn Mūsā al-Khwārizmī wrote kitāb al-ḥisāb al-hindī ("Book of Indian computation") and kitab al-jam' wa'l-tafriq al-ḥisāb al-hindī ("Addition and subtraction in Indian arithmetic"). In the early 12th century, Latin translations of these texts involving the Hindu–Arabic numeral system and arithmetic appeared, for example Liber Alghoarismi de practica arismetrice, attributed to John of Seville, and Liber Algoritmi de numero Indorum, attributed to Adelard of Bath. Here, alghoarismi or algoritmi is the Latinization of Al-Khwarizmi's name; the text starts with the phrase Dixit Algoritmi, or "Thus spoke Al-Khwarizmi".

 
The word algorism in English came to mean the use of place-value notation in calculations; it occurs in the Ancrene Wisse from circa 1225. By the time Geoffrey Chaucer wrote The Canterbury Tales in the late 14th century, he used a variant of the same word in describing augrym stones, stones used for place-value calculation. In the 15th century, under the influence of the Greek word ἀριθμός (arithmos, "number"; cf. "arithmetic"), the Latin word was altered to algorithmus. By 1596, this form of the word was used in English, as algorithm, by Thomas Hood.


== Definition ==

One informal definition is "a set of rules that precisely defines a sequence of operations", which would include all computer programs, and any bureaucratic procedure
or cook-book recipe. In general, a program is an algorithm only if it stops eventually. Formally, algorithm is an explicit set of instructions to produce an output, that can be followed by a computer or a human performing specific operations on symbols..


== History ==


=== Ancient algorithms ===
Step-by-step procedures for solving mathematical problems have been recorded since antiquity. This includes in Babylonian mathematics (around 2500 BC), Egyptian mathematics (around 1550 BC), Indian mathematics (around 800 BC and later), the Ifa Oracle (around 500 BC), Greek mathematics (around 240 BC), Chinese mathematics (around 200 BC and later), and Arabic mathematics (around 800 AD).
The earliest evidence of algorithms is found in ancient Mesopotamian mathematics. A Sumerian clay tablet found in Shuruppak near Baghdad and dated to c. 2500 BC describes the earliest division algorithm. During the Hammurabi dynasty c. 1800 – c. 1600 BC, Babylonian clay tablets described algorithms for computing formulas. Algorithms were also used in Babylonian astronomy. Babylonian clay tablets describe and employ algorithmic procedures to compute the time and place of significant astronomical events.
Algorithms for arithmetic are also found in ancient Egyptian mathematics, dating back to the Rhind Mathematical Papyrus c. 1550 BC. Algorithms were later used in ancient Hellenistic mathematics. Two examples are the Sieve of Eratosthenes, which was described in the Introduction to Arithmetic by Nicomachus, and the Euclidean algorithm, which was first described in Euclid's Elements (c. 300 BC).Examples of ancient Indian mathematics included the Shulba Sutras, the Kerala School, and the Brāhmasphuṭasiddhānta.
In the 9th century, Muḥammad ibn Mūsā al-Khwārizmī revolutionized the field by establishing the algorithm as a systematic, finite sequence of logical steps to solve mathematical problems. In his influential work, The Compendious Book on Calculation by Completion and Balancing, he moved beyond specific numerical solutions to introduce general procedures for algebraic reduction and balancing. This transformed mathematics into a 'mechanical' process of well-defined rules—a fundamental shift that laid the groundwork for modern algorithmic theory. The Latin translation of his arithmetic treatise, titled Algoritmi de numero Indorum, led to the term algorithm being derived from the Latinization of his name, Algoritmi, specifically to describe this new rule-based approach to mathematics. 
The first cryptographic algorithm for deciphering encrypted code was developed by Al-Kindi, a 9th-century Arab mathematician, in A Manuscript On Deciphering Cryptographic Messages. He gave the first description of cryptanalysis by frequency analysis, the earliest codebreaking algorithm.


=== Computers ===


==== Weight-driven clocks ====
Weight-driven clocks were a key European invention in Middle Ages, specifically the verge escapement mechanism producing the tick of mechanical clocks. Accurate automatic machines led to mechanical automata in the 13th century and computational machines—the difference and analytical engines of Charles Babbage and Ada Lovelace in the mid-19th century. Lovelace designed the first algorithm intended for a computer, Babbage's analytical engine, the first real Turing-complete computer, more than the mechanical calculators of the time. Although the full implementation of Babbage's second device was only built decades after her lifetime, Lovelace has been called "history's first programmer".


==== Electromechanical relay ====
The Jacquard loom, a precursor to punch cards, and telephone switching machines led to the development of the first computers. By the mid-19th century, the telegraph, was in use throughout the world. By the late 19th century, ticker tape (c. 1870s) and punch cards (c. 1890) were developed. Then came the teleprinter (c. 1910) with its punched-paper use of Baudot code on tape.
Telephone-switching networks of electromechanical relays were invented in 1835. These led to the invention of the digital adding device by George Stibitz in 1937. While working in Bell Laboratories, he observed the "burdensome" use of mechanical calculators with gears, prompting him to create an experimental digital adder at home.


=== Formalization ===

In 1928, a partial formalization of the modern concept of algorithms began with attempts to solve David Hilbert's Entscheidungsproblem (decision problem). Later formalizations were framed as attempts to define "effective calculability" or "effective method". Those formalizations included the Gödel–Herbrand–Kleene recursive functions of 1930, 1934 and 1935, Alonzo Church's lambda calculus of 1936, Emil Post's Formulation 1 of 1936, and Alan Turing's Turing machines of 1936–37 and 1939.


=== Modern Algorithms ===

For decades, it was assumed that algorithm evolution progresses from heuristics to formal algorithms. A Symbolic integration provides a classic illustration. In 1961, James Slagle’s program SAINT used heuristics to solve 52 of 54 freshman calculus exercises from an MIT textbook (≈96%). In 1967, Larry Moses’s SIN refined the heuristics and achieved 100% success, though it remained heuristic. Finally, in 1969, Robert Risch introduced the Risch Algorithm with formal guarantees. This trajectory defined the traditional path: heuristics evolving until a definitive, guaranteed algorithm emerged.
However, the rise of transformer-based AI has inverted this sequence — classical algorithms are now being displaced by heuristics once again.
Algorithms have evolved and improved in many ways as time goes on. Common uses of algorithms today include social media apps like Instagram and YouTube. Algorithms are used as a way to analyze what people like and push more of those things to the people who interact with them. Quantum computing uses quantum algorithm procedures to solve problems faster. More recently, in 2024, NIST updated their post-quantum encryption standards, which includes new encryption algorithms to enhance defenses against attacks using quantum computing.


== Representations ==
Algorithms can be expressed in many kinds of notation, including natural languages, pseudocode, flowcharts, drakon-charts, programming languages or control tables. Natural language expressions of algorithms tend to be verbose and ambiguous and are rarely used for complex or technical algorithms. Pseudocode, flowcharts, drakon-charts, and control tables are structured expressions of algorithms that avoid common ambiguities of natural language. Programming languages are primarily for expressing algorithms in a computer-executable form but are also used to define or document algorithms.


=== Turing machines ===
There are many possible representations and Turing machine programs can be expressed as a sequence of machine tables (see finite-state machine, state-transition table, and control table for more), as flowcharts and drakon-charts (see state diagram for more), as a form of rudimentary machine code or assembly code called "sets of quadruples", and more. Algorithm representations can also be classified into three accepted levels of Turing machine description: high-level description, implementation description, and formal description. A high-level description describes the qualities of the algorithm itself, ignoring how it is implemented on the Turing machine. An implementation description describes the general manner in which the machine moves its head and stores data to carry out the algorithm, but does not give exact states. In the most detail, a formal description gives the exact state table and list of transitions of the Turing machine.


=== Flowchart representation ===
A flowchart is a graphical aid that describes and documents an algorithm. It has four primary symbols: arrows showing program flow, rectangles (SEQUENCE, GOTO), diamonds representing decisions, and dots (OR-tie). Sub-structures can "nest" in rectangles, but only if a single exit occurs from the superstructure.


== Algorithmic analysis ==

It is often important to know the time, storage, or other cost an algorithm may require. Methods have been developed to analyse algorithms to estimate these needs. For example, an algorithm that adds up the elements of a list of n numbers would have a time requirement of ⁠
  
    
      
        O
        (
        n
        )
      
    
    {\displaystyle O(n)}
  
⁠, using big O notation. The algorithm only needs to remember two values: the sum of all the elements so far, and its current position in the input list. If the space required to store the input numbers is not counted, it has a space requirement of ⁠
  
    
      
        O
        (
        1
        )
      
    
    {\displaystyle O(1)}
  
⁠, otherwise ⁠
  
    
      
        O
        (
        n
        )
      
    
    {\displaystyle O(n)}
  
⁠ is required.
Different algorithms may complete the same task with a different set of instructions in less or more time, space, or 'effort' than others. For example, a binary search algorithm (with cost ⁠
  
    
      
        O
        (
        log
        ⁡
        n
        )
      
    
    {\displaystyle O(\log n)}
  
⁠) outperforms a sequential search (cost ⁠
  
    
      
        O
        (
        n
        )
      
    
    {\displaystyle O(n)}
  
⁠ ) when used for table lookups on sorted lists.


=== Formal versus empirical ===

The analysis, and study of algorithms is a discipline of computer science. Algorithms are often studied abstractly, without referencing a specific programming language or implementation. Like other mathematical disciplines, it focuses on the algorithm's properties, not implementation. Pseudocode is typical for analysis as it is a simple and general representation. Most algorithms are implemented on particular hardware/software platforms and their algorithmic efficiency is tested using real code. The efficiency of a particular algorithm may be insignificant for many "one-off" problems but it can be critical for algorithms designed for fast, interactive, commercial, or long-life scientific usage. Increasing the input size often exposes inefficient algorithms that are otherwise benign.
Empirical testing is useful for uncovering unexpected interactions that affect performance. Benchmarks may be used to compare before/after potential improvements to an algorithm after program optimization.
Empirical tests cannot fully replace formal analysis, and are difficult to perform fairly.


=== Execution efficiency ===

To illustrate the potential improvements possible even in well-established algorithms, a recent significant innovation, relating to FFT algorithms used for image processing, can decrease processing time up to 1,000 times for medical imaging. In general, speed improvements depend on special properties of the problem, which are very common in practical applications.


=== Best Case and Worst Case ===

The best case of an algorithm refers to the scenario or input for which the algorithm or data structure takes the least time and resources to complete its tasks.  The worst case of an algorithm is the case that causes the algorithm or data structure to consume the maximum period of time and computational resources.


== Design ==

Algorithm design may take advantage of many different approaches, such as divide-and-conquer or dynamic programming. Techniques for designing and implementing algorithms are also called algorithm design patterns. Examples include the template method pattern and the decorator pattern. An important aspect of algorithm design is efficient use of resources such as memory or time; the big O notation is used to describe how resource use changes as the size of inputs increase.


=== Structured programming ===
Any algorithm can be computed by any Turing complete model. Turing completeness only requires four instruction types—conditional GOTO, unconditional GOTO, assignment, HALT. Tausworthe augments the three Böhm-Jacopini canonical structures: SEQUENCE, IF-THEN-ELSE, and WHILE-DO, with two more: DO-WHILE and CASE. An additional benefit of a structured program is that it lends itself to proofs of correctness using mathematical induction.


== Legal status ==

By themselves, algorithms are not usually patentable. In the United States, a claim consisting solely of simple manipulations of abstract concepts, numbers, or signals does not constitute "processes", so algorithms are not patentable (as in Gottschalk v. Benson). However, practical applications of algorithms can be patentable. For example, in Diamond v. Diehr, the application of a simple feedback algorithm to aid in the curing of synthetic rubber was deemed patentable. The patenting of software is controversial, and there are criticized patents involving algorithms, especially data compression algorithms, such as Unisys's LZW patent. Additionally, some cryptographic algorithms have export restrictions (see export of cryptography).


== Classification ==


=== By implementation ===
Recursion
A recursive algorithm invokes itself repeatedly until meeting a termination condition and are a common functional programming technique. Iterative algorithms use repetitions such as loops or data structures like stacks to solve problems. Problems may be suited for one implementation or the other. The Tower of Hanoi is a puzzle commonly solved using recursive implementation. Every recursive version has an equivalent (but possibly more or less complex) iterative version, and vice versa.
Serial, parallel or distributed
Algorithms are usually discussed with the assumption that computers execute one instruction of an algorithm at a time on serial computers. Serial algorithms are designed for these environments, unlike parallel or distributed algorithms. Parallel algorithms take advantage of computer architectures where multiple processors can work on a problem at the same time. Distributed algorithms use multiple machines connected via a computer network. Parallel and distributed algorithms divide the problem into sub-problems and collect the results back together. Resource consumption in these algorithms is not only processor cycles on each processor but also the communication overhead between the processors. Some sorting algorithms can be parallelized efficiently, but their communication overhead is expensive. Iterative algorithms are generally parallelizable, but some problems have no parallel algorithms and are called inherently serial problems.
Deterministic or non-deterministic
Deterministic algorithms solve the problem with exact decisions at every step; whereas non-deterministic algorithms solve problems via guessing. Guesses are typically made more accurate through the use of heuristics.
Exact or approximate
While many algorithms reach an exact solution, approximation algorithms seek an approximation that is close to the true solution. Such algorithms have practical value for many hard problems. For example, the Knapsack problem, where there is a set of items, and the goal is to pack the knapsack to get the maximum total value. Each item has some weight and some value. The total weight that can be carried is no more than some fixed number X. So, the solution must consider the weights of items as well as their value.
Quantum algorithm
Quantum algorithms run on a realistic model of quantum computation. The term is usually used for those algorithms that seem inherently quantum or use some essential feature of Quantum computing such as quantum superposition or quantum entanglement.


=== By design paradigm ===
Another way of classifying algorithms is by their design methodology or paradigm. Some common paradigms are:

Brute-force or exhaustive search
Brute force is a problem-solving method of systematically trying every possible option until the optimal solution is found. This approach can be very time-consuming, testing every possible combination of variables. It is often used when other methods are unavailable or too complex. Brute force can solve a variety of problems, including finding the shortest path between two points and cracking passwords.
Divide and conquer
A divide-and-conquer algorithm repeatedly reduces a problem to one or more smaller instances of itself (usually recursively) until the instances are small enough to solve easily. Merge sorting is an example of divide and conquer, where an unordered list is repeatedly split into smaller lists, which are sorted in the same way and then merged. In a simpler variant of divide and conquer called prune and search or decrease-and-conquer algorithm, which solves one smaller instance of itself, and does not require a merge step. An example of a prune and search algorithm is the binary search algorithm.
Search and enumeration
Many problems (such as playing chess) can be modelled as problems on graphs. A graph exploration algorithm specifies rules for moving around a graph and is useful for such problems. This category also includes search algorithms, branch and bound enumeration, and backtracking.
Randomized algorithm
Such algorithms make some choices randomly (or pseudo-randomly). They find approximate solutions when finding exact solutions may be impractical (see heuristic method below). For some problems, the fastest approximations must involve some randomness. Whether randomized algorithms with polynomial time complexity can be the fastest algorithm for some problems is an open question known as the P versus NP problem. There are two large classes of such algorithms:
Monte Carlo algorithms return a correct answer with high probability. E.g. RP is the subclass of these that run in polynomial time.
Las Vegas algorithms always return the correct answer, but their running time is only probabilistically bound, e.g. ZPP.
Reduction of complexity
This technique transforms difficult problems into better-known problems solvable with (hopefully) asymptotically optimal algorithms. The goal is to find a reducing algorithm whose complexity is not dominated by the resulting reduced algorithms. For example, one selection algorithm finds the median of an unsorted list by first sorting the list (the expensive portion), and then pulling out the middle element in the sorted list (the cheap portion). This technique is also known as transform and conquer.
Back tracking
In this approach, multiple solutions are built incrementally and abandoned when it is determined that they cannot lead to a valid full solution.


=== Optimization problems ===
For optimization problems there is a more specific classification of algorithms; an algorithm for such problems may fall into one or more of the general categories described above as well as into one of the following:

Linear programming
When searching for optimal solutions to a linear function bound by linear equality and inequality constraints, the constraints can be used directly to produce optimal solutions. There are algorithms that can solve any problem in this category, such as the popular simplex algorithm. Problems that can be solved with linear programming include the maximum flow problem for directed graphs. If a problem also requires that any of the unknowns be integers, then it is classified in integer programming. A linear programming algorithm can solve such a problem if it can be proved that all restrictions for integer values are superficial, i.e., the solutions satisfy these restrictions anyway. In the general case, a specialized algorithm or an algorithm that finds approximate solutions is used, depending on the difficulty of the problem.
Dynamic programming
When a problem shows optimal substructures—meaning the optimal solution can be constructed from optimal solutions to subproblems—and overlapping subproblems, meaning the same subproblems are used to solve many different problem instances, a quicker approach called dynamic programming avoids recomputing solutions. For example, Floyd–Warshall algorithm, the shortest path between a start and goal vertex in a weighted graph can be found using the shortest path to the goal from all adjacent vertices. Dynamic programming and memoization go together. Unlike divide and conquer, dynamic programming subproblems often overlap. The difference between dynamic programming and simple recursion is the caching or memoization of recursive calls. When subproblems are independent and do not repeat, memoization does not help; hence dynamic programming is not applicable to all complex problems. Using memoization dynamic programming reduces the complexity of many problems from exponential to polynomial.
The greedy method
Greedy algorithms, similarly to a dynamic programming, work by examining substructures, in this case not of the problem but of a given solution. Such algorithms start with some solution and improve it by making small modifications. For some problems, they always find the optimal solution but for others they may stop at local optima. The most popular use of greedy algorithms is finding minimal spanning trees of graphs without negative cycles. Huffman Tree, Kruskal, Prim, Sollin are greedy algorithms that can solve this optimization problem.
The heuristic method
In optimization problems, heuristic algorithms find solutions close to the optimal solution when finding the optimal solution is impractical. These algorithms get closer and closer to the optimal solution as they progress. In principle, if run for an infinite amount of time, they will find the optimal solution. They can ideally find a solution very close to the optimal solution in a relatively short time. These algorithms include local search, tabu search, simulated annealing, and genetic algorithms. Some, like simulated annealing, are non-deterministic algorithms while others, like tabu search, are deterministic. When a bound on the error of the non-optimal solution is known, the algorithm is further categorized as an approximation algorithm.


== Examples ==

One of the simplest algorithms finds the largest number in a list of numbers of random order. Finding the solution requires looking at every number in the list. From this follows a simple algorithm, which can be described in plain English as:
High-level description:

If a set of numbers is empty, then there is no highest number.
Assume the first number in the set is the largest.
For each remaining number in the set: if this number is greater than the current largest, it becomes the new largest.
When there are no unchecked numbers left in the set, consider the current largest number to be the largest in the set.
(Quasi-)formal description:
Written in prose but much closer to the high-level language of a computer program, the following is the more formal coding of the algorithm in pseudocode or pidgin code:


== AI-assisted algorithm discovery ==

Artificial intelligence systems have been used to discover and optimize algorithms. In 2023, Google DeepMind introduced AlphaDev, a reinforcement learning system based on AlphaZero that discovered improved sorting and hashing algorithms. In a paper published in Nature, AlphaDev was reported to have discovered small sorting algorithms that outperformed previously known human benchmarks and were integrated into the LLVM standard C++ sorting library.
In 2025, Google DeepMind introduced AlphaEvolve, an evolutionary coding agent powered by large language models for general-purpose algorithm discovery and optimization. AlphaEvolve uses language models to propose code changes, automated evaluators to test candidate solutions, and an evolutionary process to improve promising algorithms over multiple iterations.


== See also ==


== Notes ==


== Bibliography ==

Zaslavsky, C. (1970). Mathematics of the Yoruba People and of Their Neighbors in Southern Nigeria. The Two-Year College Mathematics Journal, 1(2), 76–99. https://doi.org/10.2307/3027363
NIST Releases First 3 Finalized Post-Quantum Encryption Standards. https://www.nist.gov/news-events/news/2024/08/nist-releases-first-3-finalized-post-quantum-encryption-standards


== Further reading ==


== External links ==

"Algorithm". Encyclopedia of Mathematics. EMS Press. 2001 [1994].
Weisstein, Eric W. "Algorithm". MathWorld.
Dictionary of Algorithms and Data Structures – National Institute of Standards and Technology
Algorithm repositories
The Stony Brook Algorithm Repository – State University of New York at Stony Brook
Collected Algorithms of the ACM – Associations for Computing Machinery
The Stanford GraphBase Archived December 6, 2015, at the Wayback Machine – Stanford University


""" * 40 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.txt", help="path to training text file")
    parser.add_argument("--model", default="model.pt", help="path to save/load the model")
    parser.add_argument("--chat-only", action="store_true", help="skip training, load existing model")
    args = parser.parse_args()

    if args.chat_only:
        if not os.path.exists(args.model):
            print(f"No saved model found at {args.model}. Run without --chat-only first to train one.")
            sys.exit(1)
        model, tokenizer = load_model(args.model)
        chat(model, tokenizer)
        return

    if not os.path.exists(args.data):
        print(f"No {args.data} found -- creating a small placeholder dataset so you can try this out.")
        print("Replace data.txt with your own text (ideally 'User: ... / Bot: ...' style dialogue) and rerun.\n")
        with open(args.data, "w") as f:
            f.write(PLACEHOLDER_DATA)

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    model, tokenizer = train_model(text, save_path=args.model)
    chat(model, tokenizer)


if __name__ == "__main__":
    main()