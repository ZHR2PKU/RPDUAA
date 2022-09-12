# coding=utf-8
# Author: Haoran Zhang
# Affiliation: Peking University
# Email: henryzhang@hsc.pku.edu.cn

'''The RPDUAA program is a virtual screening tool for the rational protein design with unnatural amino acids.'''

print('*' * 80)
print('{0:*^80}'.format(' Rational Protein Design with Unnatural Amino Acids (RPDUAA, version 1.0) '))
print('*' * 80)
print('Function: Prediction of High-Confidence Sites for UAA Substitutions on a Protein')
print('Author: Haoran Zhang    Tutor: Prof. Qing Xia     Affiliation: Peking University')
print('It is strongly recommended to use a monospaced font, such as Courier or Consolas')

import os
import ssl
import numpy
import pandas
import seaborn
import warnings
import Bio.PDB
import Bio.Seq
import Bio.SeqIO
import Bio.pairwise2
import matplotlib.pyplot as plt
from scipy import stats
from Bio.Blast import NCBIXML
from Bio.Blast import NCBIWWW
from urllib.request import urlcleanup
from urllib.request import urlretrieve
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale as PCA_scale

pdb_path = './proteins/'
uaa_path = './residues/'
pre_path = './predictions/'
aa3to1 = {"ALA": 'A',  # alanine
          "CYS": 'C',  # cystine
          "ASP": 'D',  # aspartate
          "GLU": 'E',  # glutamate
          "PHE": 'F',  # phenylalanine
          "GLY": 'G',  # glycine
          "HIS": 'H',  # histidine
          "ILE": 'I',  # isoleucine
          "LYS": 'K',  # lysine
          "LEU": 'L',  # leucine
          "MET": 'M',  # methionine
          "ASN": 'N',  # asparagine
          "PRO": 'P',  # proline
          "GLN": 'Q',  # glutamine
          "ARG": 'R',  # arginine
          "SER": 'S',  # serine
          "THR": 'T',  # threonine
          "VAL": 'V',  # valine
          "TRP": 'W',  # tryptophan
          "TYR": 'Y',  # tyrosine
          }  # 20 amino acids


def welcome():
    '''This is a welcome with some basic checks.'''
    if not os.path.exists('dssp.exe'):
        dssplink = 'https://swift.cmbi.umcn.nl/gv/dssp/HTML/DSSPCMBI.EXE'
        urlcleanup()
        try:
            urlretrieve(dssplink, 'dssp.exe')
        except BaseException:
            print('-' * 80)
            print('Failed to download dssp.exe for Windows. Try it manually.')
            input('See distributions in: https://swift.cmbi.umcn.nl/gv/dssp/')
    check1 = os.path.exists('./predictions') & os.path.exists('./proteins') & os.path.exists('./residues')
    check2 = os.path.exists('./residues/known_uaa_sites.csv')
    check3 = os.path.exists('./residues/mole_properties.csv')
    check4 = os.path.exists('./residues/similarity_matrix.csv')
    check5 = os.path.exists('./dssp.exe') & os.path.exists('./Guide.pdf') & os.path.exists('./RPDUAA.exe')
    if not (check1 & check2 & check3 & check4 & check5):
        print('-' * 80)
        print('The working folder should normally contain the following subfolders and files:')
        print('\t./predictions/')
        print('\t./proteins/')
        print('\t./residues/')
        print('\t    known_uaa_sites.csv')
        print('\t    mole_properties.csv')
        print('\t    similarity_matrix.csv')
        print('\t./dssp.exe    (necessary just for protein analyses)')
        print('\t./Guide.pdf   (optional guidebook for this program)')
        print('\t./RPDUAA.exe  (optional if you use its source code)')
        input('Please prepare them (see details in Guide.pdf) and restart the RPDUAA program.')


def site_get_entropy(fastaseqs, path_xml):
    '''This function handles fasta and xml files.'''
    aas = {'A': 'alanine',
           'B': 'aspartate/asparagine',
           'C': 'cystine',
           'D': 'aspartate',
           'E': 'glutamate',
           'F': 'phenylalanine',
           'G': 'glycine',
           'H': 'histidine',
           'I': 'isoleucine',
           'K': 'lysine',
           'L': 'leucine',
           'M': 'methionine',
           'N': 'asparagine',
           'P': 'proline',
           'Q': 'glutamine',
           'R': 'arginine',
           'S': 'serine',
           'T': 'threonine',
           'U': 'selenocysteine',
           'V': 'valine',
           'W': 'tryptophan',
           'X': 'any',
           'Y': 'tyrosine',
           'Z': 'glutamate/glutamine',
           '*': 'translation stop',
           '-': 'gap of indeterminate length',
           '_': 'complemented start or end',
           }  # Amino acids
    sientropy = []
    for fa in fastaseqs:
        # Step 1
        fa_aln = []
        with open(path_xml) as xml_aligns:
            for record in NCBIXML.parse(xml_aligns):
                for alignment in record.alignments:
                    for hsp in alignment.hsps:
                        if fa.seq.find(hsp.query.replace('-', '')) != -1:
                            temp = []
                            ruler = fa.seq.split(hsp.query.replace('-', ''), 1)
                            for i in range(len(ruler[0])):
                                temp.append('_')
                            for i in range(len(hsp.query)):
                                if hsp.query[i] != '-':
                                    temp.append(hsp.sbjct[i])
                            for i in range(len(ruler[1])):
                                temp.append('_')
                            fa_aln.append(temp)
        aln_df = pandas.DataFrame(fa_aln, columns=list(fa.seq))

        # Step 2
        df_counts = pandas.DataFrame(index=aln_df.columns)
        for aa in aas:
            temp = pandas.DataFrame((aln_df == aa).sum(axis=0))
            df_counts[aa] = temp
        if aln_df.shape[0] < 0.5:
            df_probab = df_counts
            for aa in df_probab.columns:
                for ab in df_probab.index:
                    if aa == ab:
                        df_probab.loc[ab, aa] = 1
        else:
            df_probab = df_counts / aln_df.shape[0]
        df_p_logp = -df_probab * df_probab.apply(numpy.log2)
        df_probab['entropy'] = df_p_logp.sum(axis=1)
        sientropy.append(df_probab)
    return sientropy


def pdb_get_resinfo(pdb_name):
    '''This function handles PDB structures.'''
    # Step 1
    path_cif = pdb_path + pdb_name + '.cif'
    path_pdb = pdb_path + pdb_name + '.pdb'
    path_fasta = pdb_path + pdb_name + '.fasta'
    path_xml = pdb_path + pdb_name + '.xml'
    path_csv = pdb_path + pdb_name + '.csv'
    path_t1pdb = pdb_path + pdb_name + 'temp.pdb'
    path_t2pdb = pdb_path + pdb_name + 'tmpi.pdb'
    if os.path.exists(path_cif):
        structure = Bio.PDB.MMCIFParser().get_structure(pdb_name, path_cif)
    else:
        structure = Bio.PDB.PDBParser().get_structure(pdb_name, path_pdb)
    fastaseqs = list(Bio.SeqIO.parse(path_fasta, 'fasta'))
    sientropy = site_get_entropy(fastaseqs, path_xml)

    # Step 2
    chainseqs = {}
    for fa in fastaseqs:
        chainids = fa.description.split('|')[1]
        if '[' not in chainids:
            for chainid in list(chainids[4:]):
                if chainid.isupper():
                    chainseqs[chainid] = fa.seq
        else:
            chainlist = []
            for chainid in list(chainids[4:]):
                if chainid.isupper():
                    chainlist.append(chainid)
                if chainid == '[':
                    chainlist.pop()
            for chainid in chainlist:
                chainseqs[chainid] = fa.seq

    # Step 3
    selchainid = chainseqs.keys()
    delchainid = [chain.id for model in structure for chain in model if chain.id not in selchainid]
    for model in structure:
        for delid in delchainid:
            del structure[model.id][delid]
    io = Bio.PDB.PDBIO()
    io.set_structure(structure)
    io.save(path_t1pdb)

    # Step 4
    resinfo = pandas.DataFrame()
    print('-' * 80)
    print('The structure (%s.cif) has %i models, %i chains, and %i residues in total:' %
          (structure.id, len(structure), len(list(structure.get_chains())), len(list(structure.get_residues()))))

    for model in structure:
        print('\tModel[%i] has %i chains:      (Analyzing...Please wait...)' % (model.id, len(model)))
        if not os.path.exists('dssp.exe'):
            print('WARNING: External dssp.exe is required for protein analysis. Please Download it.')
            print('DSSP executable (dssp.exe): https://swift.cmbi.umcn.nl/gv/dssp/HTML/distrib.html')
            print('Reference to Bio.PDB.DSSP: https://biopython.org/docs/1.75/api/Bio.PDB.DSSP.html')
            return pandas.DataFrame()
        io.set_structure(model)
        io.save(path_t2pdb)
        hseb = Bio.PDB.HSExposureCB(model)
        dssp = Bio.PDB.DSSP(model, path_t2pdb)

        for chain in model:
            linkid = {}
            pps = Bio.PDB.Polypeptide.PPBuilder().build_peptides(chain)
            print('\t\tChain[%s] has %i residues in %i polypeptide segments:' % (chain.id, len(chain), len(pps)))
            chainseq = chainseqs[chain.id]
            skip_seq = 0
            for pp in pps:
                ppseq = pp.get_sequence()
                print('\t\t\t', pp)
                if chainseq[skip_seq:].find(ppseq) == -1:
                    ppalign0 = Bio.pairwise2.align.localxx(ppseq, chainseq[skip_seq:])[0]
                    ppalignA = ppalign0.seqA
                    if ppalign0.score < len(ppseq) * 0.95:
                        print('\t\t\tWARNING: Poorly matched sequences between fasta and cif files')
                    ppshiftN = []
                    for i in range(len(ppalignA)):
                        if ppalignA[i] != '-':
                            ppshiftN.append(ppalignA[:(i + 1)].count('-'))
                    mf_shift = 0
                    maxcount = 0
                    for j in set(ppshiftN):
                        if ppshiftN.count(j) > maxcount:
                            maxcount = ppshiftN.count(j)
                            mf_shift = skip_seq + j
                    skip_seq = mf_shift + len(ppseq)
                else:
                    mf_shift = skip_seq + chainseq[skip_seq:].find(ppseq)
                    skip_seq = mf_shift + len(ppseq)
                for rid, residue in enumerate(pp):
                    fid = residue.full_id
                    uid = str(rid + mf_shift + 1)
                    saa = aa3to1.get(residue.resname)
                    linkid[fid] = '_'.join([fid[0], str(fid[1]), fid[2], uid, saa])

            # Step 4.1
            chain_se = pandas.DataFrame()
            for se in sientropy:
                se_seq = ''.join(se.index)
                if chainseqs[chain.id] == se_seq:
                    chain_se = se.copy()
            uniqueid = []
            for i, aa in enumerate(chain_se.index):
                tempid = '_'.join([structure.id, str(model.id), chain.id, str(i + 1), aa])
                uniqueid.append(tempid)
            tempdf = pandas.DataFrame({
                'pdbname': structure.id,
                'model': model.id,
                'chain': chain.id,
                'uniqueid': uniqueid,
                'aa': chain_se.index})
            chain_se.index = tempdf.index
            tempdfse = pandas.concat([tempdf, chain_se], axis=1)

            # Step 4.2
            chaindf = pandas.DataFrame()
            for residue in chain:
                hsebres = hseb.property_dict.get(residue.full_id[2:])
                dsspres = dssp.property_dict.get(residue.full_id[2:])
                if hsebres == None:
                    hsebres = (0, 0, 0)
                if dsspres == None:
                    dsspres = (None, None, None, None, None, None, None, None, None, None, None, None, None, None)
                tempres = pandas.DataFrame({
                    'pdbname': structure.id,
                    'model': model.id,
                    'chain': chain.id,
                    'resseq': residue.id[1],
                    'resname': residue.resname,
                    'aapdb': aa3to1.get(residue.resname),
                    'hsebup': hsebres[0],
                    'hsebdn': hsebres[1],
                    'hseb': hsebres[0] + hsebres[1],
                    'aadssp': dsspres[1],
                    'ss': dsspres[2],
                    'rasa': dsspres[3],
                    'uniqueid': linkid.get(residue.full_id)}, index=[0])
                chaindf = chaindf.append(tempres)
            chaindf = chaindf.merge(tempdfse, on=['pdbname', 'model', 'chain', 'uniqueid'], how='right')
            resinfo = resinfo.append(chaindf, ignore_index=True)

    # Step 5
    splitid = resinfo['uniqueid'].str.split('_', expand=True)
    resinfo['site'] = splitid[4] + splitid[3]
    resinfo = resinfo.drop(columns='uniqueid')
    while True:
        try:
            resinfo.to_csv(path_csv)
            break
        except BaseException:
            input('WARNING: Please make sure the csv file is closed and type in "OK" here: ')
    print('The protein (%s) has been analyzed. See report in: %s' % (pdb_name, path_csv))
    print('-' * 80)
    os.remove(path_t1pdb)
    os.remove(path_t2pdb)
    return resinfo


def uaa_get_uaainfo():
    '''This function handles the similarity matrix.'''
    similarity_mat = pandas.read_csv(uaa_path + 'similarity_matrix.csv')
    similarity_uaa = similarity_mat.iloc[0:20, 21:]
    similarity_uaa.index = [a.split('_')[1] for a in similarity_mat['canvas'][0:20]]
    return similarity_uaa


def download3files(pdb_name):
    path_cif = pdb_path + pdb_name + '.cif'
    path_fasta = pdb_path + pdb_name + '.fasta'
    path_xml = pdb_path + pdb_name + '.xml'
    print('===>>> Preparing the cif file of %s ===>>>' % pdb_name)
    if os.path.exists(path_cif):
        print('%s.cif already exists in the folder (%s).' % (pdb_name, pdb_path))
    else:
        print('Downloading the %s.cif file...Please wait a few seconds...' % pdb_name)
        ciflink = 'https://files.rcsb.org/download/' + pdb_name.upper() + '.cif'
        urlcleanup()
        try:
            urlretrieve(ciflink, path_cif)
        except BaseException:
            print('Failed to download %s.cif from PDB. Please check and download it manually.' % pdb_name)
            return
        if os.path.exists(path_cif):
            print('%s.cif has been successfully downloaded to the folder (%s).' % (pdb_name, pdb_path))
        else:
            print('Failed to download %s.cif from PDB. Please check and download it manually.' % pdb_name)
            return
    print('===>>> Preparing the fasta file of %s ===>>>' % pdb_name)
    if os.path.exists(path_fasta):
        print('%s.fasta already exists in the folder (%s).' % (pdb_name, pdb_path))
    else:
        print('Downloading the %s.fasta file...Please wait a few seconds...' % pdb_name)
        fastalink = 'https://www.pdbus.org/fasta/entry/' + pdb_name.upper()
        urlcleanup()
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            urlretrieve(fastalink, path_fasta)
        except BaseException:
            print('Failed to download %s.fasta from PDB. Please check and download it manually.' % pdb_name)
        with open(path_fasta) as fastafile:
            failurefasta = fastafile.read()
            if 'No valid PDB ID' not in failurefasta:
                print('%s.fasta has been successfully downloaded to the folder (%s).' % (pdb_name, pdb_path))
            else:
                os.remove(path_fasta)
                print('Failed to download %s.fasta from PDB. Please check and download it manually.' % pdb_name)
                return
    print('===>>> Preparing the xml file of %s ===>>>' % pdb_name)
    if os.path.exists(path_xml):
        print('%s.xml already exists in the folder (%s).' % (pdb_name, pdb_path))
    else:
        print('Downloading the %s.xml file...Please wait a few minutes...' % pdb_name)
        with open(path_fasta) as fastafile:
            fasta_records = fastafile.read()
        try:
            result_handle = NCBIWWW.qblast('blastp', 'nr', fasta_records, expect=0.05, hitlist_size=100)
        except BaseException:
            print('Failed to download %s.xml because the NCBI website blocked this BLAST query.' % pdb_name)
            return
        with open(path_xml, 'w') as save_file:
            save_file.write(result_handle.read())
        result_handle.close()
        if os.path.exists(path_xml):
            print('%s.xml has been successfully downloaded to the folder (%s).' % (pdb_name, pdb_path))
        else:
            print('Failed to download %s.xml from NCBI. Please check and BLAST it manually.' % pdb_name)
            return
    if os.path.exists(path_cif) & os.path.exists(path_fasta) & os.path.exists(path_xml):
        print('=== All 3 files (cif, fasta, xml) of %s are ready in the folder (%s).' % (pdb_name, pdb_path))
        return True


def preparemolpro(df, abs_diff=False):
    df['uaasimi'] = [similarity_uaa.loc[df['aa'][i], df['uaaname'][i]] for i in df.index]
    # Step 1
    df['uaaMW'] = df['uaa'].map(mole_properties['MW'])
    df['uaaAlogP'] = df['uaa'].map(mole_properties['AlogP'])
    df['uaaHBA'] = df['uaa'].map(mole_properties['HBA'])
    df['uaaHBD'] = df['uaa'].map(mole_properties['HBD'])
    df['uaaRB'] = df['uaa'].map(mole_properties['RB'])
    df['uaaPSA'] = df['uaa'].map(mole_properties['PSA'])
    df['uaaEstate'] = df['uaa'].map(mole_properties['Estate'])
    df['uaaPolar'] = df['uaa'].map(mole_properties['Polar'])
    # Step 2
    df['aaMW'] = df['aa'].map(mole_properties['MW'])
    df['aaAlogP'] = df['aa'].map(mole_properties['AlogP'])
    df['aaHBA'] = df['aa'].map(mole_properties['HBA'])
    df['aaHBD'] = df['aa'].map(mole_properties['HBD'])
    df['aaRB'] = df['aa'].map(mole_properties['RB'])
    df['aaPSA'] = df['aa'].map(mole_properties['PSA'])
    df['aaEstate'] = df['aa'].map(mole_properties['Estate'])
    df['aaPolar'] = df['aa'].map(mole_properties['Polar'])
    # Step 3
    if abs_diff:
        df['diffMW'] = abs(df['uaaMW'] / df['aaMW'] - 1)
        df['diffAlogP'] = abs(df['uaaAlogP'] - df['aaAlogP'])
        df['diffHBA'] = abs(df['uaaHBA'] - df['aaHBA'])
        df['diffHBD'] = abs(df['uaaHBD'] - df['aaHBD'])
        df['diffRB'] = abs(df['uaaRB'] - df['aaRB'])
        df['diffPSA'] = abs(df['uaaPSA'] - df['aaPSA'])
        df['diffEstate'] = abs(df['uaaEstate'] - df['aaEstate'])
        df['diffPolar'] = abs(df['uaaPolar'] - df['aaPolar'])
    else:
        df['diffMW'] = df['uaaMW'] / df['aaMW'] - 1
        df['diffAlogP'] = df['uaaAlogP'] - df['aaAlogP']
        df['diffHBA'] = df['uaaHBA'] - df['aaHBA']
        df['diffHBD'] = df['uaaHBD'] - df['aaHBD']
        df['diffRB'] = df['uaaRB'] - df['aaRB']
        df['diffPSA'] = df['uaaPSA'] - df['aaPSA']
        df['diffEstate'] = df['uaaEstate'] - df['aaEstate']
        df['diffPolar'] = df['uaaPolar'] - df['aaPolar']
    # Step 4
    df[['resname', 'aapdb', 'aadssp', 'ss']] = df[['resname', 'aapdb', 'aadssp', 'ss']].fillna('-')
    df[['hsebup', 'hsebdn', 'hseb']] = df[['hsebup', 'hsebdn', 'hseb']].fillna(0)
    df[['rasa']] = df[['rasa']].fillna(1)
    df['sshelix'] = df['ss'].isin(['H', 'G', 'I'])
    df['ssstrand'] = df['ss'].isin(['E'])
    df['ssturn'] = df['ss'].isin(['T'])
    df['cdamber'] = df['codon'].isin(['TAG', 'UAG'])
    df['cdochre'] = df['codon'].isin(['TAA', 'UAA'])
    df['cdopal'] = df['codon'].isin(['TGA', 'UGA'])
    df['proofd'] = df['proof'].isin(['direct'])
    df['proofi'] = df['proof'].isin(['indirect'])
    global varX
    varX = ['entropy', 'uaasimi', 'rasa', 'hsebup', 'hsebdn']
    # Step 5
    varX.extend(['diffAlogP', 'diffEstate', 'diffPSA', 'diffPolar', 'diffHBA', 'diffHBD', 'diffRB', 'diffMW'])
    varX.extend(['sshelix', 'ssstrand', 'ssturn', 'cdamber', 'cdochre', 'cdopal', 'proofd', 'proofi'])
    return df


def function1():
    '''This function corresponds to Main Menu [1].'''
    print('\nTo analyze a protein, you should prepare 3 files in the folder (%s):' % pdb_path)
    print('> Download the cif and fasta files of a protein from PDB (https://www.rcsb.org)')
    print('> BLAST the fasta file on NCBI (blastp) and download all results into a xml file')
    print('> Rename all 3 files with their PDB name (4 characters, lowercase, e.g., 6mh2)\n')

    # Step 1
    all_pro_files = {a for a in os.listdir(pdb_path)}
    all_pro_names = {os.path.splitext(a)[0] for a in os.listdir(pdb_path)}
    ready_pro_names = []
    unanalyzed_pro_names = []
    for i in all_pro_names:
        pdb_3set1 = {'%s.cif' % i, '%s.fasta' % i, '%s.xml' % i}
        pdb_3set2 = {'%s.pdb' % i, '%s.fasta' % i, '%s.xml' % i}
        if pdb_3set1.issubset(all_pro_files) or pdb_3set2.issubset(all_pro_files):
            ready_pro_names.append(i)
            if not os.path.exists(pdb_path + i + '.csv'):
                unanalyzed_pro_names.append(i)
    ready_pro_names.sort()
    print('Currently prepared proteins in the folder (%s) or available PDB names:' % pdb_path, end='')
    for i in range(len(ready_pro_names)):
        if i % 9 == 0:
            print()
        print('\t%s' % ready_pro_names[i], end='')
    if bool(unanalyzed_pro_names):
        print('\nCurrently unanalyzed proteins include:', unanalyzed_pro_names, end='')

    # Step 2
    while True:
        input_name = str.lower(input('\nPlease input the PDB name of the protein that you want to analyze here: '))
        if input_name.startswith('_'):
            input_name = input_name[1:]
            input_down = download3files(input_name)
            if bool(input_down):
                break
            else:
                return
        if input_name.lower() == 'cancel':
            return
        pdb_3set1 = {'%s.cif' % input_name, '%s.fasta' % input_name, '%s.xml' % input_name}
        pdb_3set2 = {'%s.pdb' % input_name, '%s.fasta' % input_name, '%s.xml' % input_name}
        if pdb_3set1.issubset(all_pro_files) or pdb_3set2.issubset(all_pro_files):
            break
        print('WARNING: {%s.cif, %s.fasta, %s.xml} NOT READY in the folder (%s)' % (
            input_name, input_name, input_name, pdb_path))
        print('To download them in RPDUAA, add a single "_" before the PDB name (e.g., _6mh2).')
        print('If you want to cancel analyzing and return to Main Menu, just type in "cancel".')
    pdb_name = input_name

    # Step 3
    path_csv = pdb_path + pdb_name + '.csv'
    if os.path.exists(path_csv):
        resinfo = pandas.read_csv(path_csv, index_col=0)
        print('-' * 80)
        print('The previously analyzed data (%s) of protein (%s) is loaded:' % (path_csv, pdb_name))
    else:
        resinfo = pdb_get_resinfo(pdb_name)
        if resinfo.empty:
            print('Your protein is not analyzed due to the lack of dssp.exe in the working folder!')
            return
        print('The freshly analyzed data (%s) of protein (%s) is loaded:' % (path_csv, pdb_name))
    print(resinfo[['pdbname', 'model', 'chain', 'resseq', 'site', 'aa', 'ss', 'rasa', 'hseb', 'entropy']])
    subtest1 = (resinfo.aapdb == resinfo.aadssp)
    subtest2 = (resinfo.aapdb == resinfo.aa)
    subtest3 = (~(subtest1 & subtest2)) & (~resinfo.resname.isna())
    sublines = subtest3.sum()
    if sublines > 0:
        print('WARNING: The csv file of %s contains %s abnormal lines as indexed below:' % (pdb_name, sublines), end='')
        for i, value in enumerate(numpy.where(subtest3 == True)[0].tolist()):
            if i % 13 == 0:
                print()
            print('{0:>6}'.format(value), end='')
        print('\nPlease CHECK these lines in the csv file and ADJUST them manually if necessary!')
    print('If you want to re-analyze %s, delete its csv file and run Main Menu [1] again.' % pdb_name)


def function2():
    '''This function corresponds to Main Menu [2].'''
    print('\nCurrently available unnatural amino acids (UAAs) in the program include:', end='')
    for i, a in enumerate(similarity_uaa.columns):
        if i % 5 == 0:
            print()
        print('{0:<16}'.format(a), end='')
    print('\nRefer to "Guide.pdf" for chemical formulae and full names of these indexed UAAs.')
    print('To add more UAAs, renew "mole_properties.csv" and "similarity_matrix.csv" files.')
    print('Should any question arise, contact the author email (henryzhang@hsc.pku.edu.cn).')


def function3():
    '''This function corresponds to Main Menu [3].'''
    known_uaa_sites = pandas.read_csv(uaa_path + 'known_uaa_sites.csv', index_col=0)
    print()
    print('From the database, %i records of experimentally verified UAA sites are loaded:' % known_uaa_sites.shape[0])
    print(known_uaa_sites.loc[:, ['pdbname', 'model', 'chain', 'site', 'aa', 'ss', 'rasa', 'entropy', 'uaa', 'succ']])
    print('> To BROWSE records, close RPDUAA and turn to (%sknown_uaa_sites.csv).' % uaa_path)
    print('> To DELETE records, just delete rows in the csv file, save and re-open RPDUAA.')
    print('> To APPEND records, keep the csv file closed (important) and follow the hints.')
    choice = input('Do you want to APPEND more records of known UAA sites to the database? (Y/N) ')
    if choice not in ['Y', 'y']:
        return

    # Step 1
    print('-' * 80)
    print('Currently available UAAs include:    (Type in "cancel" to cancel appending)', end='')
    for i, a in enumerate(similarity_uaa.columns):
        if i % 5 == 0:
            print()
        print('{0:<16}'.format(a), end='')
    print()
    temp_uaa = ''
    while True:
        choice = input('Please select a listed UAA and input its id/name (e.g., UA22 or NAEK): ')
        if choice.lower() == 'cancel':
            return
        for a in similarity_uaa.columns:
            if choice in a.split('_') or choice == a:
                temp_uaa = a
        if bool(temp_uaa):
            break
    print('You have selected %s as the substitute UAA for upcoming records.' % temp_uaa)

    # Step 2
    print('-' * 80)
    print('Currently available proteins include:    (Type in "cancel" to cancel appending)', end='')
    csvlist = [a for a in os.listdir(pdb_path) if a.endswith('.csv')]
    csvlist.sort()
    for i, a in enumerate(csvlist):
        if i % 10 == 0:
            print()
        print('{0:<8}'.format(a.split('.')[0]), end='')
    print()
    temp_csv = ''
    while True:
        choice = input('Please select a listed protein and input its pdb name (e.g., 6mh2): ')
        choice = choice.lower()
        if choice.lower() == 'cancel':
            return
        for a in csvlist:
            if choice in a.split('.') or choice == a:
                temp_csv = a
        if bool(temp_csv):
            break
    pdb_name = temp_csv.split('.')[0]
    resinfo = pandas.read_csv(pdb_path + temp_csv, index_col=0)
    print('You have selected %s as the base protein for upcoming records.' % pdb_name)

    # Step 3
    modellist = set(resinfo.model)
    if len(modellist) == 1:
        temp_model = 0
    else:
        print('-' * 80)
        print('Current protein %s has multiple models:' % pdb_name, modellist)
        temp_model = 0
        while True:
            choice = input('Please select a listed model and input its id (e.g., 0 or 1): ')
            if choice.lower() == 'cancel':
                return
            if choice.isdigit():
                if int(choice) in modellist:
                    temp_model = int(choice)
                    break

    # Step 4
    resinfo_sub = resinfo[resinfo.model == temp_model]
    chainlist = list(set(resinfo_sub.chain))
    chainlist.sort()
    print('-' * 80)
    print('Current model of %s has the following chains:' % pdb_name, chainlist)
    temp_chain = ''
    while True:
        choice = input('Please select a listed chain and input its id (e.g., A): ')
        if choice.lower() == 'cancel':
            return
        for a in chainlist:
            if choice == a:
                temp_chain = a
        if bool(temp_chain):
            break
    print('You have selected [%s][model %i][chain %s] for upcoming records.' % (pdb_name, temp_model, temp_chain))

    # Step 5
    print('-' * 80)
    print('Current chain has the following sites:    (Type in "cancel" to cancel appending)', end='')
    resinfo_sub = resinfo_sub[resinfo_sub.chain == temp_chain]
    sitelist = list(resinfo_sub.site)
    for i, a in enumerate(sitelist):
        if i % 10 == 0:
            print()
        print('{0:<8}'.format(a), end='')
    success_sites = input('\nPlease input your success UAA sites from the above list, separated by a space:\n')
    if success_sites.lower() == 'cancel':
        return
    success_sites = success_sites.upper().split(' ')
    success_sites_in = [a for a in success_sites if a in sitelist]
    success_sites_ig = [a for a in success_sites if a not in sitelist]
    print('In-list success sites:', success_sites_in)
    print('Ignored success sites:', success_sites_ig)
    failure_sites = input('Please input your failure UAA sites from the above list, separated by a space:\n')
    if failure_sites.lower() == 'cancel':
        return
    failure_sites = failure_sites.upper().split(' ')
    failure_sites_in = [a for a in failure_sites if a in sitelist]
    failure_sites_ig = [a for a in failure_sites if a not in sitelist]
    print('In-list failure sites:', failure_sites_in)
    print('Ignored failure sites:', failure_sites_ig)
    ref_effi = []
    for temp_site in success_sites_in + failure_sites_in:
        while True:
            try:
                a = input('Input the UAA incorporation efficiency (e.g., 0.2835) for %s site: ' % temp_site)
                if a.lower() == 'cancel':
                    return
                if bool(a):
                    a = float(a)
                else:
                    a = None
                ref_effi.append(a)
                break
            except BaseException:
                print('Please input a float number (e.g., 0.2835), or simply input nothing.')
    ref_codon = input('Please input the expanded codon (e.g., TAG, TAA or TGA) for these sites: ')
    print('-' * 80)
    ref_doi = input('Please input a reference DOI or any note for these sites: ')
    ref_year = input('Please input the year (e.g., 2019) of the reference: ')
    print('Abbreviations of methods used to prove UAA incorporations are listed below:')
    print('(1) MS: Mass Spectrometry, such as MALDI-TOF, ESI-MS, LC/MS')
    print('(2) GEL: Electrophoresis Gels, such as Western blot, SDS-PAGE, etc.')
    print('(3) OPT: Optical Methods, such as luminescence, fluorescence, FRET, etc.')
    print('(4) FUN: Protein Functions, such as enzymatic activity, ion channel activity.')
    print('(5) VP: Virus Package, such as CPE, virus titer.      (6) YIE: Protein Yield.')
    print('(7) DCA: Synthesize dCA-UAA and link to tRNA, usually used in cell-free systems.')
    print('(8) AXT: Auxotrophic Strains, replacing a natural amino acid in medium with UAA.')
    print('(9) NMR: Nuclear Magnetic Resonance            (10) XRAY: X-Ray crystallography.')
    ref_method = input('Please select methods by inputting abbreviations (e.g., MS+GEL+OPT): ')
    print('-' * 80)
    print('Finally, you may evaluate the reference as direct, indirect, or other proofs:')
    print('> Direct proofs: UAA incorporation proved by MS+GEL results or valid methods.')
    print('> Indirect proofs: UAA incorporation proved by protein functions or viruses.')
    print('> Other proofs: Additional methods (DCA or AXT) beyond genetic code expansion.')
    ref_proof = input('Please input your proof evaluation (e.g., direct, indirect, or other): ')

    # Step 6
    print('-' * 80)
    resinfo2 = resinfo[(resinfo.pdbname == pdb_name) & (resinfo.model == temp_model) & (resinfo.chain == temp_chain)]
    success_sites_df = resinfo2[resinfo2.site.isin(success_sites_in)]
    failure_sites_df = resinfo2[resinfo2.site.isin(failure_sites_in)]
    success_sites_df['succ'] = 1
    failure_sites_df['succ'] = 0
    new_site_info = success_sites_df.append(failure_sites_df, ignore_index=True)
    add_info_name = ['refdoi', 'year', 'proof', 'method', 'codon', 'uaaname', 'uaa', 'succ', 'efficiency']
    new_site_info['refdoi'] = ref_doi
    new_site_info['year'] = ref_year
    new_site_info['proof'] = ref_proof
    new_site_info['method'] = ref_method
    new_site_info['uaaname'] = temp_uaa
    new_site_info['uaa'] = temp_uaa.split('_')[1]
    new_site_info['codon'] = ref_codon
    new_site_info['efficiency'] = pandas.Series(ref_effi)
    print('You have prepared new records of experimentally verified UAA sites as below:')
    print(new_site_info[['pdbname', 'chain', 'site', 'entropy', 'rasa', 'year', 'codon', 'uaa', 'succ', 'efficiency']])
    confirm = input('Do you want to add these new records to the database (Y/N): ')
    if (confirm.lower() == 'cancel') or (confirm.lower() == 'n'):
        print('The new records above were not appended to the database because you canceled.')
        return
    known_uaa_sites = known_uaa_sites.append(new_site_info, ignore_index=True)
    known_uaa_sites = known_uaa_sites.loc[:, :'site'].join(known_uaa_sites.loc[:, add_info_name])
    while True:
        try:
            known_uaa_sites.to_csv(uaa_path + 'known_uaa_sites.csv')
            break
        except BaseException:
            input('WARNING: Please make sure the csv file is closed and type in "OK" here: ')
    print('New records of known UAA sites have been successfully appended to the database.')


def function4(test_inlab_data=False, use_efficiency=False, abs_diff=False):
    '''This function corresponds to Main Menu [4] Predict High-Confidence Sites for UAA Substitutions.'''
    known_uaa_sites = pandas.read_csv(uaa_path + 'known_uaa_sites.csv', index_col=0)
    if use_efficiency:
        effic_uaa_sites = known_uaa_sites[~known_uaa_sites.efficiency.isna()]
        effic_uaa_sites = effic_uaa_sites[effic_uaa_sites.method.str.contains('YIE')]
        # effic_uaa_sites = effic_uaa_sites[effic_uaa_sites.efficiency < 1]
        effic_n = effic_uaa_sites.shape[0]
        effic_uaa_sites = preparemolpro(effic_uaa_sites, abs_diff=abs_diff)
        effic_X = effic_uaa_sites[varX]
        effic_y = effic_uaa_sites['efficiency']
        '''glm_formula = 'log((efficiency+Min)/(Max-efficiency)) = intercept + k1*V1 + k2*V2 + ... + kn*Vn'
        Maxy = 4.5  # Just like Logistic Regression. Be ware of log(E/(1-E)) and NaN.
        Miny = 0.001
        effic_trany = numpy.log((effic_y + Miny) / (Maxy - effic_y))  # Pre-processing of efficiency for GLM
        def eff_reverse(v):
            vr = (Maxy * numpy.exp(v) - Miny) / (1 + numpy.exp(v))
            vr[vr < 0.000001] = 0.000001
            return vr'''

        glm_formula = 'log(efficiency+1) = intercept + k1*V1 + k2*V2 + ... + kn*Vn'
        effic_trany = numpy.log(effic_y + 1)  # Pre-processing of efficiency for GLM

        def eff_reverse(v):
            vr = numpy.exp(v) - 1
            vr[vr < 0.000001] = 0.000001
            return vr

        '''glm_formula = 'log(efficiency+1) = intercept + k1*V1 + k2*V2 + ... + kn*Vn'
        effic_trany = numpy.log(effic_y + 1)  # Pre-processing of efficiency for GLM
        eff_reverse = lambda v: numpy.exp(v) - 1  # Reverse transformation after GLM'''
        effic_model = LinearRegression()
        effic_model.fit(effic_X, effic_trany)
        effic_predy = effic_model.predict(effic_X)
        effic_z = eff_reverse(effic_predy)
        effic_r = numpy.corrcoef(effic_y, effic_z)[0, 1]
        effic_b = numpy.mean(abs(effic_y - effic_z))
        plt.figure(1, figsize=(8, 4.8), dpi=85, tight_layout=True)
        plt.scatter(effic_y, effic_z)
        k, b = numpy.polyfit(effic_y, effic_z, 1)
        fitline = k * effic_y + b
        plt.plot(effic_y, fitline, 'k-')
        plt.title('UAA Incorporation Efficiency Predictied by a Generalized Linear Model (Exact Yield)')
        plt.xlabel('Detected UAA incorporation efficiency (%d records from literature with exact yield)' % effic_n)
        plt.ylabel('Predicted UAA incorporation efficiency by RPDUAA')
        plt.text(effic_y.max(), effic_z.min(), glm_formula, ha='right')
        plt.text(effic_y.max(), effic_z.min() + 0.1, "Average deviation of prediction = %.4f" % effic_b, ha='right')
        plt.text(effic_y.max(), effic_z.min() + 0.2, "Pearson correlation coefficient = %.4f" % effic_r, ha='right')
        plt.show()
    if test_inlab_data:
        inlab_uaa_sites = known_uaa_sites[~known_uaa_sites.refdoi.str.startswith('10.')]
        if inlab_uaa_sites.shape[0] == 0:
            print('ERROR: No records of prospective laboratory data in the database!!!')
            return
    known_uaa_sites = known_uaa_sites[known_uaa_sites.refdoi.str.startswith('10.')]
    print('\nThe prediction will use the Database of Experimentally Verified UAA Sites.')
    print('> Subset 1: All records in the database (proof=direct/indirect/other)  ****')
    print('> Subset 2: Records from miscellaneous results (proof=direct/indirect)  ***')
    print('> Subset 3: Records strongly supported by MS+GEL results (proof=direct)   *')
    print('> Subset 4: Balanced records (>300 proofs; equal successes and failures) **')
    print('> Subset 5: Non-predicted PDB records (>1000 proofs; solved structures) ***')
    print('> Subset 6: Non-redundant records (drop duplicate proofs in literature)  **')
    print('> Subset 7: Non-viral protein records (excluding those viral proteins)    *')
    print('> Subset 8: Viral protein records (those demonstrated by virus package)   *')
    print('> Subset 9: Amber suppression records (the expanded codon is TAG or UAG) **')
    print('Suffixes: d/f for dynamic/fixed cutoff; h/t for holdout/timesplit validation.')
    choice_range = ['0', 'cancel', 'p1e']
    choice_range1 = ['1', '1d', '1f', '1h', '1t', '1hd', '1hf', '1dh', '1fh', '1td', '1tf', '1dt', '1ft', '1x']
    choice_range2 = ['2', '2d', '2f', '2h', '2t', '2hd', '2hf', '2dh', '2fh', '2td', '2tf', '2dt', '2ft', '2x']
    choice_range3 = ['3', '3d', '3f', '3h', '3t', '3hd', '3hf', '3dh', '3fh', '3td', '3tf', '3dt', '3ft', '3x']
    choice_range4 = ['4', '4d', '4f', '4h', '4t', '4hd', '4hf', '4dh', '4fh', '4td', '4tf', '4dt', '4ft', '4x']
    choice_range5 = ['5', '5d', '5f', '5h', '5t', '5hd', '5hf', '5dh', '5fh', '5td', '5tf', '5dt', '5ft', '5x']
    choice_range6 = ['6', '6d', '6f', '6h', '6t', '6hd', '6hf', '6dh', '6fh', '6td', '6tf', '6dt', '6ft', '6x']
    choice_range7 = ['7', '7d', '7f', '7h', '7t', '7hd', '7hf', '7dh', '7fh', '7td', '7tf', '7dt', '7ft', '7x']
    choice_range8 = ['8', '8d', '8f', '8h', '8t', '8hd', '8hf', '8dh', '8fh', '8td', '8tf', '8dt', '8ft', '8x']
    choice_range9 = ['9', '9d', '9f', '9h', '9t', '9hd', '9hf', '9dh', '9fh', '9td', '9tf', '9dt', '9ft', '9x']
    choice_range.extend(choice_range1)
    choice_range.extend(choice_range2)
    choice_range.extend(choice_range3)
    choice_range.extend(choice_range4)
    choice_range.extend(choice_range5)
    choice_range.extend(choice_range6)
    choice_range.extend(choice_range7)
    choice_range.extend(choice_range8)
    choice_range.extend(choice_range9)
    while True:
        choice = input('Please select a subset (1, 2, or 5) from the above list for prediction: ')
        choice = choice.lower()
        if choice in choice_range:
            break
        else:
            print('Improper selection! Try the listed numbers or type in "cancel" to exit.')
    if (choice == '0') or (choice == 'cancel'):
        return
    elif choice in choice_range1:
        subset_title = 'Records from the Whole Database'
    elif choice in choice_range2:
        subset_title = 'Records (Indirect/Direct Proofs)'
        known_uaa_sites = known_uaa_sites[known_uaa_sites.proof.isin(['direct', 'indirect'])]
    elif choice in choice_range3:
        subset_title = 'Records (Direct MS+GEL Proofs)'
        known_uaa_sites = known_uaa_sites[known_uaa_sites.proof.isin(['direct'])]
    elif choice in choice_range4:
        subset_title = 'Balanced Records (S:F=1:1)'
        fail_records = known_uaa_sites[known_uaa_sites.succ == 0]
        succ_records = known_uaa_sites[known_uaa_sites.succ == 1]
        numpy.random.seed(9)
        succ_records = succ_records.sample(fail_records.shape[0])
        known_uaa_sites = fail_records.append(succ_records)
    elif choice in choice_range5:
        subset_title = 'Non-Predicted PDB Records'
        known_uaa_sites = known_uaa_sites[known_uaa_sites.pdbname.str.get(0).str.isdigit()]
    elif choice in choice_range6:
        subset_title = 'Non-Redundant (NR) Records'
        known_uaa_sites = known_uaa_sites.drop_duplicates(['pdbname', 'model', 'chain', 'site', 'uaaname', 'succ'])
    elif choice in choice_range7:
        subset_title = 'Non-Viral Protein Records'
        known_uaa_sites = known_uaa_sites[~known_uaa_sites.method.str.contains('VP')]
    elif choice in choice_range8:
        subset_title = 'Viral Protein Records'
        known_uaa_sites = known_uaa_sites[known_uaa_sites.method.str.contains('VP')]
    elif choice in choice_range9:
        subset_title = 'Amber Suppression Records'
        known_uaa_sites = known_uaa_sites[known_uaa_sites.codon.isin(['TAG', 'UAG'])]

    # Step 1
    known_uaa_sites = preparemolpro(known_uaa_sites, abs_diff=abs_diff)
    my_X = known_uaa_sites[varX]
    my_y = known_uaa_sites['succ']
    if choice == 'p1e':
        my_model = LogisticRegression(C=1e9)
        my_model.fit(my_X, my_y)
        known_uaa_sites = pandas.read_csv(uaa_path + 'known_uaa_sites.csv', index_col=0)
        known_uaa_sites = preparemolpro(known_uaa_sites, abs_diff=abs_diff)
        my_X = known_uaa_sites[varX]
        my_p = my_model.predict_proba(my_X)[:, 1]
        my_p = pandas.Series(my_p, name='probability', index=known_uaa_sites.index)
        my_pred = known_uaa_sites.join(my_p)
        if use_efficiency:
            my_z = eff_reverse(effic_model.predict(my_X))
            my_z = pandas.Series(my_z, name='pred_effi', index=my_X.index)
            my_pred = my_pred.join(my_z)
        my_pred.to_csv(pre_path + 'baseprob.csv')
        print('The whole database has been used for prediction (%sbaseprob.csv).' % pre_path)
        return

    # Step 2
    if 'x' in choice:
        x100_report = pandas.DataFrame(columns=['rand_state', 'opti_cutoff', 'accuracy80', 'roc_auc'])
        for curr in range(0, 100):
            numpy.random.seed(curr)
            if choice != '4x':
                validation_method = 'holdout'
                output_filestring = 'hovd'
                X_train, X_test, y_train, y_test = train_test_split(my_X, my_y, test_size=0.2)
            else:
                known_uaa_sites = pandas.read_csv(uaa_path + 'known_uaa_sites.csv', index_col=0)
                known_uaa_sites = known_uaa_sites[known_uaa_sites.refdoi.str.startswith('10.')]
                fail_records = known_uaa_sites[known_uaa_sites.succ == 0]
                succ_records = known_uaa_sites[known_uaa_sites.succ == 1]
                succ_records = succ_records.sample(fail_records.shape[0])
                known_uaa_sites = fail_records.append(succ_records)
                known_uaa_sites = preparemolpro(known_uaa_sites, abs_diff=abs_diff)
                my_X = known_uaa_sites[varX]
                my_y = known_uaa_sites['succ']
                validation_method = 'resampling'
                output_filestring = 'bsrs'
                X_train, X_test, y_train, y_test = my_X, my_X, my_y, my_y
            # Step 2.1
            my_model = LogisticRegression(C=1e9)
            my_model.fit(X_train, y_train)
            y_pred = my_model.predict_proba(X_test)[:, 1]
            y_pred = pandas.Series(y_pred, index=y_test.index)
            # Step 2.2
            opti_cutoff2 = 0.5
            opti_table2 = numpy.array([])
            opti_score2 = 0
            opti_split2 = 0
            for my_cutoff in range(0, 100):
                my_table = confusion_matrix(y_test, y_pred > my_cutoff / 100)
                my_score = (my_table[0, 0] + my_table[1, 1]) / my_table.sum()
                specificity = my_table[0, 0] / (my_table[0, 0] + my_table[0, 1])
                sensitivity = my_table[1, 1] / (my_table[1, 0] + my_table[1, 1])
                my_split = specificity + sensitivity
                if my_split >= opti_split2:
                    opti_cutoff2 = my_cutoff / 100
                    opti_table2 = my_table
                    opti_score2 = my_score
                    opti_split2 = my_split
            fpr, tpr, threshold = roc_curve(y_test, my_model.decision_function(X_test))
            roc_auc = auc(fpr, tpr)
            print(curr, '%.2f' % opti_cutoff2, '%.14f' % opti_score2, '%.14f' % roc_auc, sep='\t')
            curr_report = pandas.DataFrame({'rand_state': curr,
                                            'opti_cutoff': opti_cutoff2,
                                            'accuracy80': opti_score2,
                                            'roc_auc': roc_auc}, index=[curr])
            x100_report = x100_report.append(curr_report)
        # Step 2.3
        plt.figure(1, figsize=(14.4, 4.8), dpi=85, tight_layout=True)
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
        plt.sca(ax1)
        plt.plot(x100_report.rand_state, x100_report.opti_cutoff)
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Random seed of %s validation' % validation_method)
        plt.ylabel('Optimal cutoff of the prediction model')
        plt.title('Optimal cutoff of 100 %s validations' % validation_method)
        plt.sca(ax2)
        plt.plot(x100_report.rand_state, x100_report.accuracy80)
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Random seed of %s validation' % validation_method)
        plt.ylabel('Accuracy at the optimal cutoff')
        plt.title('Accuracy of 100 %s validations' % validation_method)
        plt.sca(ax3)
        plt.plot(x100_report.rand_state, x100_report.roc_auc)
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Random seed of %s validation' % validation_method)
        plt.ylabel('Area under the ROC curve')
        plt.title('ROC curve area of 100 %s validations' % validation_method)
        while True:
            try:
                x100_report.to_csv(pre_path + ('%sx100.csv' % output_filestring), index=False)
                break
            except BaseException:
                input('WARNING: Please close the %sx100.csv file and type in "OK" here: ' % output_filestring)
        print('Close the figure and see full reports in (%s%sx100.csv).' % (pre_path, output_filestring))
        plt.show()
        return

    # Step 3
    elif 'h' in choice:
        numpy.random.seed()
        X_train, X_test, y_train, y_test = train_test_split(my_X, my_y, test_size=0.2)
        print('\nYou have activated the holdout validation for the prediction model.')
        print('Training data: randomly picked 80% records, used to train the model parameters.')
        print('Testing data: the rest 20% records, used to plot the figures for evaluations.')
    elif 't' in choice:
        yearlist = [str(i) for i in range(2002, 2222)]
        while True:
            startyear = input('\nPlease input the starting year (e.g., 2019) of testing data: ')
            if startyear in yearlist:
                startyear = int(startyear)
                break
            elif (startyear == '0') | (startyear == 'cancel'):
                return
            else:
                print('Improper starting year! It must be a number between 2002 and 2021.')
        X_train = my_X[known_uaa_sites.year < startyear]
        y_train = my_y[known_uaa_sites.year < startyear]
        X_test = my_X[known_uaa_sites.year >= startyear]
        y_test = my_y[known_uaa_sites.year >= startyear]
        print('Training data (publication year < %s): %s records.' % (startyear, y_train.shape[0]))
        print('Testing data (publication year >= %s): %s records.' % (startyear, y_test.shape[0]))
        if y_train.shape[0] < 100:
            print('ERROR: Training data have <100 records, not enough for machine learning.')
            return
        if y_test.shape[0] < 100:
            print('ERROR: Testing data have <100 records, not enough for model validation.')
            return
    else:
        X_train, X_test, y_train, y_test = my_X, my_X, my_y, my_y
    if test_inlab_data:
        subset_title = 'Unpublished Laboratory Records'
        inlab_uaa_sites = preparemolpro(inlab_uaa_sites, abs_diff=abs_diff)
        X_test = inlab_uaa_sites[varX]
        y_test = inlab_uaa_sites['succ']

    # Step 4
    my_model = LogisticRegression(C=1e9)
    my_model.fit(X_train, y_train)
    y_pred = my_model.predict_proba(X_test)[:, 1]
    y_pred = pandas.Series(y_pred, index=y_test.index)
    if 'd' in choice:
        cutoff_dummy = sum(known_uaa_sites.succ == 1) / len(known_uaa_sites)
        cutoff_range = int(cutoff_dummy * 100)
        cutoff_range = range(cutoff_range, cutoff_range + 1)
        print('\nYou have chosen the dynamic cutoff for the predicted probability.')
        print('The dynamic cutoff = successes / (successes + failures) = %.2f.' % cutoff_dummy)
    elif 'f' in choice:
        while True:
            try:
                fixed_cutoff = input('\nPlease input your fixed cutoff (e.g., 0.80) of the predicted probability: ')
                fixed_cutoff = int(float(fixed_cutoff) * 100)
                if (fixed_cutoff < 1) or (fixed_cutoff > 99):
                    raise BaseException
                break
            except BaseException:
                print('ERROR! Please input a valid cutoff (e.g., 0.80 or 0.51) between 0.01 and 0.99.')
        print('You have manually fixed the cutoff to %.2f for the predicted probability.' % (fixed_cutoff / 100))
        cutoff_range = range(fixed_cutoff, fixed_cutoff + 1)
    else:
        cutoff_range = range(0, 100)
    opti_cutoff = 0.5
    opti_table = numpy.array([])
    opti_score = 0
    opti_split = 0
    for my_cutoff in cutoff_range:
        my_table = confusion_matrix(y_test, y_pred > my_cutoff / 100)
        my_score = (my_table[0, 0] + my_table[1, 1]) / my_table.sum()
        specificity = my_table[0, 0] / (my_table[0, 0] + my_table[0, 1])
        sensitivity = my_table[1, 1] / (my_table[1, 0] + my_table[1, 1])
        my_split = specificity + sensitivity
        if my_split >= opti_split:
            opti_cutoff = my_cutoff / 100
            opti_table = my_table
            opti_score = my_score
            opti_split = my_split
    print('\nAbout the prediction model:')
    print('> GOAL: To predict the probability of successful UAA substitution on a protein.')
    print('> METHOD: Perform logistic regression on the above database of known UAA sites.')
    print('> FORMULA: logit(P) = intercept + k1 * entropy + k2 * uaasimi + k3 * rasa + ...')
    print('> By setting a probability cutoff of %.2f, the accuracy of the model is %.2f%%.' %
          (opti_cutoff, opti_score * 100))
    print('> The confusion matrix of the prediction model at this optimized cutoff is:')
    print(opti_table)

    # Step 5
    plt.figure(1, figsize=(14.4, 4.8), dpi=85, tight_layout=True)
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    pltdata = pandas.DataFrame()
    pltdata['succfail'] = y_test
    pltdata['predprob'] = y_pred
    grouped = pltdata.groupby('succfail')
    q25 = grouped['predprob'].quantile(0.25)
    q50 = grouped['predprob'].quantile(0.5)
    q75 = grouped['predprob'].quantile(0.75)
    numpy.random.seed(1)
    fluct = numpy.random.randn(len(y_test)) / 10
    fluct = pandas.Series(fluct, index=y_test.index)
    fspan = numpy.sqrt(sum(y_test == 0) / sum(y_test == 1))
    bin10count0, bin11edge0 = numpy.histogram(y_pred[y_test == 0])
    bin10count1, bin11edge1 = numpy.histogram(y_pred[y_test == 1])
    bin10scale0 = bin10count0 / bin10count0.max()
    bin10scale1 = bin10count1 / bin10count1.max()
    for i in y_test.index:
        if y_test[i] == 0:
            for j in range(len(bin10scale0)):
                if (bin11edge0[j] <= y_pred[i] <= bin11edge0[j + 1]):
                    fluct[i] = fluct[i] * bin10scale0[j] * fspan
                    # fluct[i] = fluct[i] * numpy.sqrt(bin10scale0[j]) * fspan
                    break
        else:
            for j in range(len(bin10scale1)):
                if (bin11edge1[j] <= y_pred[i] <= bin11edge1[j + 1]):
                    fluct[i] = fluct[i] * bin10scale1[j]
                    # fluct[i] = fluct[i] * numpy.sqrt(bin10scale1[j])
                    break
    bar = 0.1
    col = 'red'
    co2 = {0: 'blue', 1: 'darkgreen'}

    # Step 6
    plt.sca(ax1)
    pca = PCA(n_components=2)
    newX = pca.fit_transform(PCA_scale(X_test))
    plt.scatter(newX[:, 0], newX[:, 1], s=10, c=y_test.map(co2))
    plt.xlabel('Principal component 1 (EVR=%.2f%%)' % (100 * pca.explained_variance_ratio_[0]))
    plt.ylabel('Principal component 2 (EVR=%.2f%%)' % (100 * pca.explained_variance_ratio_[1]))
    plt.title('Principal Component Analysis (PCA) of Features')
    '''tsvd = TruncatedSVD(n_components=2)  # Alternative tsvd method for dimension reduction
    newX = tsvd.fit_transform(X_test)
    plt.scatter(newX[:, 0], newX[:, 1], s=10, c=y_test.map(co2))
    plt.xlabel('Principal component 1 (EVR=%.2f%%)' % (100 * tsvd.explained_variance_ratio_[0]))
    plt.ylabel('Principal component 2 (EVR=%.2f%%)' % (100 * tsvd.explained_variance_ratio_[1]))
    plt.title('Truncated Singular Value Decomposition (TSVD)')'''
    '''from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    tsne.fit_transform(PCA_scale(X_test))
    newX = tsne.embedding_
    plt.scatter(newX[:, 0], newX[:, 1], s=10, c=y_test.map(co2))
    plt.title('TSNE Embedding of Features')'''

    # Step 7
    plt.sca(ax2)
    plt.scatter(y_test + fluct, y_pred, s=10, c=y_test.map(co2))
    plt.hlines(q25[0], xmin=0 - bar, xmax=0 + bar, color=col)
    plt.hlines(q50[0], xmin=0 - 2 * bar, xmax=0 + 2 * bar, color=col)
    plt.hlines(q75[0], xmin=0 - bar, xmax=0 + bar, color=col)
    plt.vlines(0, q25[0], q75[0], color=col)
    plt.hlines(q25[1], xmin=1 - bar, xmax=1 + bar, color=col)
    plt.hlines(q50[1], xmin=1 - 2 * bar, xmax=1 + 2 * bar, color=col)
    plt.hlines(q75[1], xmin=1 - bar, xmax=1 + bar, color=col)
    plt.vlines(1, q25[1], q75[1], color=col)
    plt.axhline(opti_cutoff, color='black', linewidth=2, alpha=0.8)
    plt.text(0.5, opti_cutoff + 0.01, 'cutoff=%s' % opti_cutoff, ha='center')
    plt.text(0.5, opti_cutoff - 0.12, 'accuracy\n=%.2f%%' % (opti_score * 100), ha='center')
    p_value = stats.mannwhitneyu(y_pred[y_test == 0], y_pred[y_test == 1]).pvalue
    p_report = ('P=%.4f' % p_value) if p_value >= 0.0001 else ('P<0.0001')
    plt.text(0.5, opti_cutoff - 0.25, p_report, ha='center')
    plt.xlim([-0.5, 1.5])
    plt.xticks([0, 1], ['Failure sites (N=%s)' % sum(y_test == 0), 'Success sites (N=%s)' % sum(y_test == 1)])
    plt.xlabel('Experimental category of UAA substitution sites (N=%s)' % len(y_test))
    plt.ylabel('Predicted probability of successful UAA substitution')
    if 'h' in choice:
        plot_title = 'HV-plot'
    elif 't' in choice:
        plot_title = 'TS-plot'
    else:
        plot_title = 'Scatters'
    plt.title('%s of %s %s' % (plot_title, X_test.shape[0], subset_title))

    # Step 8
    plt.sca(ax3)
    fpr, tpr, threshold = roc_curve(y_test, my_model.decision_function(X_test))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area=%.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate (1 - Specificity)')
    plt.ylabel('True positive rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    print('(Close the Figure to START Predicting)')
    plt.show()

    # Step 9
    print('\nThe RPDUAA program provides three strategies of prediction or optimization:')
    print('(1) For a given protein and a given UAA, find the optimal substitution sites.')
    print('(2) For a given protein and given substitution site(s), find the optimal UAAs.')
    print('(3) Full probability matrix for all UAAs scanning all sites of a given protein.')
    while True:
        strategy = input('Please input your choice (1, 2 or 3) here or type in "cancel" to exit: ')
        if strategy.lower() in ['cancel', '0', '1', '2', '3']:
            break
        else:
            print('Improper choice! Please try the listed numbers (1, 2 or 3) again.')
    if (strategy.lower() == 'cancel') or (strategy == '0'):
        return

    # Step 10
    elif strategy == '1':
        # Step 10.1
        print('-' * 80)
        print('Currently available proteins include:    (Type in "cancel" to cancel predicting)', end='')
        csvlist = [a for a in os.listdir(pdb_path) if a.endswith('.csv')]
        csvlist.sort()
        for i, a in enumerate(csvlist):
            if i % 10 == 0:
                print()
            print('{0:<8}'.format(a.split('.')[0]), end='')
        print()
        temp_csv = ''
        while True:
            choice = input('Please select a listed protein and input its pdb name (e.g., 6mh2): ')
            choice = choice.lower()
            if choice.lower() == 'cancel':
                return
            for a in csvlist:
                if choice in a.split('.') or choice == a:
                    temp_csv = a
            if bool(temp_csv):
                break
        pdb_name = temp_csv.split('.')[0]
        resinfo = pandas.read_csv(pdb_path + temp_csv, index_col=0)
        resinfo = resinfo[resinfo.aa.isin(aa3to1.values())]
        print('You have selected %s as the base protein for upcoming predictions.' % pdb_name)

        # Step 10.2
        print('-' * 80)
        print('Currently available UAAs include:    (Type in "cancel" to cancel predicting)', end='')
        for i, a in enumerate(similarity_uaa.columns):
            if i % 5 == 0:
                print()
            print('{0:<16}'.format(a), end='')
        print()
        temp_uaa = ''
        while True:
            choice = input('Please select a listed UAA and input its id/name (e.g., UA22 or NAEK): ')
            if choice.lower() == 'cancel':
                return
            for a in similarity_uaa.columns:
                if choice in a.split('_') or choice == a:
                    temp_uaa = a
            if bool(temp_uaa):
                break
        print('You have selected %s as the substitute UAA for upcoming predictions.' % temp_uaa)

        # Step 10.3
        print('-' * 80)
        temp_codon = ''
        print('Your UAA incorporation may adopt one of the expanded codons as listed below:')
        print('(1) Amber: TAG/UAG;   (2) Ochre: TAA/UAA;   (3) Opal: TGA/UGA;   (4) Others.')
        while True:
            choice = input('Please select a listed condon by inputting its number (e.g., 1, 2, 3 or 4): ')
            if choice.lower() == 'cancel':
                return
            if choice in ['1', '2', '3', '4']:
                if choice == '1':
                    temp_codon = 'TAG'
                elif choice == '2':
                    temp_codon = 'TAA'
                elif choice == '3':
                    temp_codon = 'TGA'
                elif choice == '4':
                    temp_codon = 'XXX'
                break
        print('You have selected %s as the expanded codon for UAA incorporation.' % temp_codon)
        print('-' * 80)
        temp_proof = ''
        print('Your UAA incorporation may be based on different experimental proofs:')
        print('(1) Looser definition: UAA incorporation can be proved by miscellaneous results.')
        print('(2) Strict definition: UAA incorporation can only be proved by MS+GEL results.')
        while True:
            choice = input('Please select a definition by inputting its number (1 more recommended): ')
            if choice.lower() == 'cancel':
                return
            if choice in ['1', '2']:
                if choice == '1':
                    temp_proof = 'indirect'
                    print('You have selected a looser definition of UAA incorporation (indirect proofs).')
                elif choice == '2':
                    temp_proof = 'direct'
                    print('You have selected a strict definition of UAA incorporation (direct proofs).')
                break
        resinfo['refdoi'] = 'Predicted_records'
        resinfo['proof'] = temp_proof
        resinfo['codon'] = temp_codon
        resinfo['uaaname'] = temp_uaa
        resinfo['uaa'] = temp_uaa.split('_')[1]
        resinfo['probability'] = 0
        resinfo = preparemolpro(resinfo, abs_diff=abs_diff)
        my_X2 = resinfo[varX]
        my_y2 = pandas.Series(my_model.predict_proba(my_X2)[:, 1], name='probability', index=resinfo.index)
        resinfo['probability'] = my_y2

        # Step 10.4
        report0 = resinfo.loc[:, :'aa'].join(resinfo.loc[:, 'entropy':'probability'])
        if use_efficiency:
            my_z2 = eff_reverse(effic_model.predict(my_X2))
            my_z2 = pandas.Series(my_z2, name='efficiency', index=resinfo.index)
            report0 = report0.join(my_z2)
        report0 = report0.sort_values(by='probability', ascending=False)
        path_pred = pre_path + 'prediction.csv'
        for i in range(10000):
            if not os.path.exists(pre_path + 'pred' + '{0:0>4}'.format(i) + '.csv'):
                path_pred = pre_path + 'pred' + '{0:0>4}'.format(i) + '.csv'
                break
        report0.to_csv(path_pred)
        print('-' * 80)
        report1 = resinfo[['pdbname', 'chain', 'site', 'ss', 'entropy', 'rasa', 'uaasimi', 'uaa', 'probability']]
        report2 = report1[report1['probability'] > opti_cutoff].sort_values(by='probability', ascending=False)
        print('The probability of successful UAA substitutions on protein %s is predicted.' % pdb_name)
        if report2.shape[0] == 0:
            print('No record of predicted successful UAA sites (whose probability > cutoff %.2f).' % opti_cutoff)
        else:
            print('The predicted successful UAA sites (whose probability > cutoff %.2f) include:' % opti_cutoff)
            print(report2)
        print('See (%s) for full details of this prediction.' % path_pred)

    # Step 11
    elif strategy == '2':
        # Step 11.1
        print('-' * 80)
        print('Currently available proteins include:    (Type in "cancel" to cancel predicting)', end='')
        csvlist = [a for a in os.listdir(pdb_path) if a.endswith('.csv')]
        csvlist.sort()
        for i, a in enumerate(csvlist):
            if i % 10 == 0:
                print()
            print('{0:<8}'.format(a.split('.')[0]), end='')
        print()
        temp_csv = ''
        while True:
            choice = input('Please select a listed protein and input its pdb name (e.g., 6mh2): ')
            choice = choice.lower()
            if choice.lower() == 'cancel':
                return
            for a in csvlist:
                if choice in a.split('.') or choice == a:
                    temp_csv = a
            if bool(temp_csv):
                break
        pdb_name = temp_csv.split('.')[0]
        resinfo = pandas.read_csv(pdb_path + temp_csv, index_col=0)
        resinfo = resinfo[resinfo.aa.isin(aa3to1.values())]
        print('You have selected %s as the base protein for upcoming predictions.' % pdb_name)

        # Step 11.2
        modellist = set(resinfo.model)
        if len(modellist) == 1:
            temp_model = 0
        else:
            print('-' * 80)
            print('Current protein %s has multiple models:' % pdb_name, modellist)
            temp_model = 0
            while True:
                choice = input('Please select a listed model and input its id (e.g., 0 or 1): ')
                if choice.lower() == 'cancel':
                    return
                if choice.isdigit():
                    if int(choice) in modellist:
                        temp_model = int(choice)
                        break

        # Step 11.3
        resinfo_sub = resinfo[resinfo.model == temp_model]
        chainlist = list(set(resinfo_sub.chain))
        chainlist.sort()
        print('-' * 80)
        print('Current model of %s has the following chains:' % pdb_name, chainlist)
        temp_chain = ''
        while True:
            choice = input('Please select a listed chain and input its id (e.g., A): ')
            if choice.lower() == 'cancel':
                return
            if choice in chainlist:
                temp_chain = choice
                break
        print('You have selected [%s][model %i][chain %s] for upcoming predictions.' % (pdb_name, temp_model,
                                                                                        temp_chain))
        # Step 11.4
        print('-' * 80)
        print('Current chain has the following sites:    (Type in "cancel" to cancel appending)', end='')
        resinfo_sub = resinfo_sub[resinfo_sub.chain == temp_chain]
        sitelist = list(resinfo_sub.site)
        for i, a in enumerate(sitelist):
            if i % 10 == 0:
                print()
            print('{0:<8}'.format(a), end='')
        interest_sites = input('\nPlease input your interest UAA sites from the list, separated by a space:\n')
        if interest_sites.lower() == 'cancel':
            return
        interest_sites = interest_sites.upper().split(' ')
        interest_sites_in = [a for a in interest_sites if a in sitelist]
        interest_sites_ig = [a for a in interest_sites if a not in sitelist]
        print('In-list interest sites:', interest_sites_in)
        print('Ignored interest sites:', interest_sites_ig)

        # Step 11.5
        print('-' * 80)
        temp_codon = ''
        print('Your UAA incorporation may adopt one of the expanded codons as listed below:')
        print('(1) Amber: TAG/UAG;   (2) Ochre: TAA/UAA;   (3) Opal: TGA/UGA;   (4) Others.')
        while True:
            choice = input('Please select a listed condon by inputting its number (e.g., 1, 2, 3 or 4): ')
            if choice.lower() == 'cancel':
                return
            if choice in ['1', '2', '3', '4']:
                if choice == '1':
                    temp_codon = 'TAG'
                elif choice == '2':
                    temp_codon = 'TAA'
                elif choice == '3':
                    temp_codon = 'TGA'
                elif choice == '4':
                    temp_codon = 'XXX'
                break
        print('You have selected %s as the expanded codon for UAA incorporation.' % temp_codon)
        print('-' * 80)
        temp_proof = ''
        print('Your UAA incorporation may be based on different experimental proofs:')
        print('(1) Looser definition: UAA incorporation can be proved by miscellaneous results.')
        print('(2) Strict definition: UAA incorporation can only be proved by MS+GEL results.')
        while True:
            choice = input('Please select a definition by inputting its number (1 more recommended): ')
            if choice.lower() == 'cancel':
                return
            if choice in ['1', '2']:
                if choice == '1':
                    temp_proof = 'indirect'
                    print('You have selected a looser definition of UAA incorporation (indirect proofs).')
                elif choice == '2':
                    temp_proof = 'direct'
                    print('You have selected a strict definition of UAA incorporation (direct proofs).')
                break
        print('-' * 80)
        print('Currently available UAAs include:', end='')
        for i, a in enumerate(similarity_uaa.columns):
            if i % 5 == 0:
                print()
            print('{0:<16}'.format(a), end='')
        print()
        resinfo['refdoi'] = 'Predicted_records'
        resinfo2 = pandas.DataFrame()
        for interest_site in interest_sites_in:
            for temp_uaa in similarity_uaa.columns:
                interest_resinfo = resinfo[(resinfo.pdbname == pdb_name) & (resinfo.model == int(temp_model))
                                           & (resinfo.chain == temp_chain) & (resinfo.site == interest_site)]
                interest_resinfo['proof'] = temp_proof
                interest_resinfo['codon'] = temp_codon
                interest_resinfo['uaaname'] = temp_uaa
                interest_resinfo['uaa'] = temp_uaa.split('_')[1]
                interest_resinfo['probability'] = 0
                interest_resinfo['uaasimi'] = interest_resinfo['aa'].map(similarity_uaa[temp_uaa])
                resinfo2 = resinfo2.append(interest_resinfo, ignore_index=True)
        resinfo2 = preparemolpro(resinfo2, abs_diff=abs_diff)
        my_X2 = resinfo2[varX]
        my_y2 = pandas.Series(my_model.predict_proba(my_X2)[:, 1], name='probability', index=resinfo2.index)
        resinfo2['probability'] = my_y2

        # Step 11.6
        report0 = resinfo2.loc[:, :'aa'].join(resinfo2.loc[:, 'entropy':'probability'])
        if use_efficiency:
            my_z2 = eff_reverse(effic_model.predict(my_X2))
            my_z2 = pandas.Series(my_z2, name='efficiency', index=resinfo2.index)
            report0 = report0.join(my_z2)
        report0 = report0.sort_values(by='probability', ascending=False)
        path_pred = pre_path + 'prediction.csv'
        for i in range(10000):
            if not os.path.exists(pre_path + 'pred' + '{0:0>4}'.format(i) + '.csv'):
                path_pred = pre_path + 'pred' + '{0:0>4}'.format(i) + '.csv'
                break
        report0.to_csv(path_pred)
        print('-' * 80)
        report1 = resinfo2[['pdbname', 'chain', 'site', 'ss', 'entropy', 'rasa', 'uaasimi', 'uaa', 'probability']]
        report2 = report1[report1['probability'] > opti_cutoff].sort_values(by='probability', ascending=False)
        print('The probability of successful UAA substitutions on protein %s is predicted.' % pdb_name)
        if report2.shape[0] == 0:
            print('No record of predicted successful UAA sites (whose probability > cutoff %.2f).' % opti_cutoff)
        else:
            print('The predicted successful UAA sites (whose probability > cutoff %.2f) include:' % opti_cutoff)
            print(report2)
        print('See (%s) for full details of this prediction.' % path_pred)

    # Step 12
    elif strategy == '3':
        # Step 12.1
        print('-' * 80)
        print('Currently available proteins include:    (Type in "cancel" to cancel predicting)', end='')
        csvlist = [a for a in os.listdir(pdb_path) if a.endswith('.csv')]
        csvlist.sort()
        for i, a in enumerate(csvlist):
            if i % 10 == 0:
                print()
            print('{0:<8}'.format(a.split('.')[0]), end='')
        print()
        temp_csv = ''
        while True:
            choice = input('Please select a listed protein and input its pdb name (e.g., 6mh2): ')
            choice = choice.lower()
            if choice.lower() == 'cancel':
                return
            for a in csvlist:
                if choice in a.split('.') or choice == a:
                    temp_csv = a
            if bool(temp_csv):
                break
        pdb_name = temp_csv.split('.')[0]
        resinfo = pandas.read_csv(pdb_path + temp_csv, index_col=0)
        resinfo = resinfo[resinfo.aa.isin(aa3to1.values())]
        print('You have selected %s as the base protein for upcoming predictions.' % pdb_name)

        # Step 12.2
        modellist = set(resinfo.model)
        if len(modellist) == 1:
            temp_model = 0
        else:
            print('-' * 80)
            print('Current protein %s has multiple models:' % pdb_name, modellist)
            temp_model = 0
            while True:
                choice = input('Please select a listed model and input its id (e.g., 0 or 1): ')
                if choice.lower() == 'cancel':
                    return
                if choice.isdigit():
                    if int(choice) in modellist:
                        temp_model = int(choice)
                        break
        resinfo = resinfo[resinfo.model == temp_model]

        # Step 12.3
        print('-' * 80)
        temp_codon = ''
        print('Your UAA incorporation may adopt one of the expanded codons as listed below:')
        print('(1) Amber: TAG/UAG;   (2) Ochre: TAA/UAA;   (3) Opal: TGA/UGA;   (4) Others.')
        while True:
            choice = input('Please select a listed condon by inputting its number (e.g., 1, 2, 3 or 4): ')
            if choice.lower() == 'cancel':
                return
            if choice in ['1', '2', '3', '4']:
                if choice == '1':
                    temp_codon = 'TAG'
                elif choice == '2':
                    temp_codon = 'TAA'
                elif choice == '3':
                    temp_codon = 'TGA'
                elif choice == '4':
                    temp_codon = 'XXX'
                break
        print('You have selected %s as the expanded codon for UAA incorporation.' % temp_codon)
        print('-' * 80)
        temp_proof = ''
        print('Your UAA incorporation may be based on different experimental proofs:')
        print('(1) Looser definition: UAA incorporation can be proved by miscellaneous results.')
        print('(2) Strict definition: UAA incorporation can only be proved by MS+GEL results.')
        while True:
            choice = input('Please select a definition by inputting its number (1 more recommended): ')
            if choice.lower() == 'cancel':
                return
            if choice in ['1', '2']:
                if choice == '1':
                    temp_proof = 'indirect'
                    print('You have selected a looser definition of UAA incorporation (indirect proofs).')
                elif choice == '2':
                    temp_proof = 'direct'
                    print('You have selected a strict definition of UAA incorporation (direct proofs).')
                break
        print('-' * 80)
        print('Currently available UAAs include:', end='')
        for i, a in enumerate(similarity_uaa.columns):
            if i % 5 == 0:
                print()
            print('{0:<16}'.format(a), end='')
        print('\nPlease wait. The prediction takes a few seconds and a full heatmap will emerge.')
        resinfo['refdoi'] = 'Predicted_records'
        resinfo3 = pandas.DataFrame()
        for temp_uaa in similarity_uaa.columns:
            interest_resinfo = resinfo[(resinfo.pdbname == pdb_name) & (resinfo.model == int(temp_model))]
            interest_resinfo['proof'] = temp_proof
            interest_resinfo['codon'] = temp_codon
            interest_resinfo['uaaname'] = temp_uaa
            interest_resinfo['uaa'] = temp_uaa.split('_')[1]
            interest_resinfo['probability'] = 0
            interest_resinfo['uaasimi'] = interest_resinfo['aa'].map(similarity_uaa[temp_uaa])
            resinfo3 = resinfo3.append(interest_resinfo, ignore_index=True)
        resinfo3 = preparemolpro(resinfo3, abs_diff=abs_diff)
        my_X3 = resinfo3[varX]
        if use_efficiency:
            my_y3 = pandas.Series(eff_reverse(effic_model.predict(my_X3)), name='efficiency', index=resinfo3.index)
        else:
            my_y3 = pandas.Series(my_model.predict_proba(my_X3)[:, 1], name='probability', index=resinfo3.index)
        resinfo3['prediction'] = my_y3

        # Step 12.4
        heatdata = resinfo3[resinfo3.uaaname == 'UA01_Anap'][['pdbname', 'model', 'chain', 'resseq', 'aa', 'site']]
        heatdata['chain_site'] = heatdata.chain + '_' + heatdata.site
        for temp_uaa in similarity_uaa.columns:
            heatdata[temp_uaa] = resinfo3[resinfo3.uaaname == temp_uaa]['prediction'].to_list()
        heatplot = heatdata.loc[:, 'UA01_Anap':]
        heatplot.index = heatdata.chain_site.to_list()
        path_pred = pre_path + 'prediction.csv'
        for i in range(10000):
            if not os.path.exists(pre_path + 'pred' + '{0:0>4}'.format(i) + '.csv'):
                path_pred = pre_path + 'pred' + '{0:0>4}'.format(i) + '.csv'
                break
        heatdata.to_csv(path_pred)
        heatplot = heatdata.loc[:, 'UA01_Anap':]
        heatplot.index = heatdata.chain_site
        heatplot.index.names = ['Chain_Site identifier for the substitution site in the target protein']
        heatplot.columns.names = ['Registered_UAA in the RPDUAA program']
        plt.figure(1, figsize=(14.4, 4.8), dpi=85, tight_layout=True)
        seaborn.heatmap(heatplot.T)
        if use_efficiency:
            plt.title('Heatmap of predicted efficiency of successful UAA substitutions in protein (%s)' % pdb_name)
            print('See (%s) for the full efficiency matrix of %s.' % (path_pred, pdb_name))
        else:
            plt.title('Heatmap of predicted probability of successful UAA substitutions in protein (%s)' % pdb_name)
            print('See (%s) for the full probability matrix of %s.' % (path_pred, pdb_name))
        plt.show()


def function5(show_database_summary=False):
    '''This function corresponds to Main Menu [5] About the RPDUAA Program and How to Cite/Use RPDUAA.'''
    # Some keywords of the RPDUAA program
    print('\nAbout the RPDUAA program:')
    print('> Full Name: Rational Protein Design with Unnatural Amino Acids')
    print('> Short Name and Version: RPDUAA (version 1.0, on 2021-10-1)')
    print('> Author and Email: Haoran Zhang (henryzhang@hsc.pku.edu.cn)')
    print('> Tutor and Email: Professor Qing Xia (xqing@hsc.pku.edu.cn)')
    print('> Contributors: Zhetao Zheng, Xuesheng Wu, Xu Yang, Haishuang Lin')
    print('> Affiliation: School of Pharmaceutical Sciences, Peking University')
    print('> Address: Xueyuan Road 38, Haidian District, Beijing 100191, China')
    # Summary of the database of known UAA sites
    if show_database_summary:
        print('\nAbout the underlying database:')
        print('> Full Name: The Database of Experimentally Verified UAA Substitution Sites')
        print('> Short Name and Date: The Database of Known UAA Sites (first release in 2021)')
        known_uaa_sites = pandas.read_csv(uaa_path + 'known_uaa_sites.csv', index_col=0)
        lab_size = known_uaa_sites[~known_uaa_sites.refdoi.str.startswith('10.')].shape[0]
        known_uaa_sites = known_uaa_sites[known_uaa_sites.refdoi.str.startswith('10.')]
        pub_size = known_uaa_sites.shape[0]
        suc_size = known_uaa_sites[known_uaa_sites.succ == 1].shape[0]
        fai_size = known_uaa_sites[known_uaa_sites.succ == 0].shape[0]
        print('> Total Size: %s published records (excluding %s unpublished in-lab records)' % (pub_size, lab_size))
        print('> Composition: %s success records and %s failure records of UAA substitution' % (suc_size, fai_size))
        eff_size = known_uaa_sites[~known_uaa_sites.efficiency.isna()].shape[0]
        yie_rows = (~known_uaa_sites.efficiency.isna()) & known_uaa_sites.method.str.contains('YIE')
        yie_size = known_uaa_sites[yie_rows].shape[0]
        print('> Efficiency: %s records with efficiency (%s records based on exact yield)' % (eff_size, yie_size))
        paper_no = len(set(known_uaa_sites.refdoi.tolist()))
        year0 = int(known_uaa_sites.year.min())
        year1 = int(known_uaa_sites.year.max())
        print('> Source: Collected from %s research articles published between %s and %s' % (paper_no, year0, year1))
        total_uaa = mole_properties.shape[0] - 20
        inuse_uaa = len(set(known_uaa_sites.uaaname.tolist()))
        print('> UAAs: %s UAAs registered in total (the database only uses %s UAAs of them)' % (total_uaa, inuse_uaa))
        pro1 = len(set(known_uaa_sites.pdbname.tolist()))
        pro2 = len(set(known_uaa_sites[known_uaa_sites.pdbname.str.get(0).str.isdigit()].pdbname.tolist()))
        pro3 = pro1 - pro2
        print('> Proteins:  %s proteins in total (%s from PDB and %s predicted structures)' % (pro1, pro2, pro3))
    # Brief description of RPDUAA and how to cite RPDUAA
    print('''\nHow to Cite or Use RPDUAA:               (Refer to "Guide.pdf" for full usages)
RPDUAA is an open-source program for rational design of proteins with unnatural
amino acids. RPDUAA can be freely used in scholar circumstances, as long as you
cite it properly. Commercial uses of RPDUAA should be authorized by authors and
Haoran Zhang holds the rights including explanation and new version maintenance.
You can cite the RPDUAA program in your articles as follows or likewise:
The unnatural amino acid (UAA) substitution sites on protein are preselected by
the RPDUAA program (version 1.0, Haoran Zhang et al, Peking University).''')


if __name__ == '__main__':
    welcome()
    warnings.filterwarnings('ignore')
    similarity_uaa = uaa_get_uaainfo()
    mole_properties = pandas.read_csv(uaa_path + 'mole_properties.csv', index_col=0)
    mole_properties.index = [i.split('_')[1] for i in mole_properties['name'].to_list()]
    while True:
        print('*' * 80)
        print('''// Main Menu:
        [1] Analyze Protein Structures (cif + fasta + xml --> csv)
        [2] Show the List of Available Unnatural Amino Acids (UAAs)
        [3] Manage the Database of Experimentally Verified UAA Sites
        [4] Predict High-Confidence Sites for UAA Substitutions
        [5] About the RPDUAA Program and How to Cite/Use RPDUAA''')
        choice = input('Please input your choice (e.g., 1, 2, 3, 4, 5, or 0 to exit) here: ')
        if choice not in ['0', '1', '2', '3', '4', '5', '4a', '4i', '4e', '5s']:
            print('Incorrect input! Only menu numbers (1, 2, 3, 4, 5, and 0) are allowed.')
            continue
        elif choice == '0':
            print('Thank you for using RPDUAA.')
            break
        elif choice == '1':
            function1()
        elif choice == '2':
            function2()
        elif choice == '3':
            function3()
        elif choice == '4':
            function4()
        elif choice == '5':
            function5()
        elif choice == '4a':
            function4(abs_diff=True)
        elif choice == '4i':
            function4(test_inlab_data=True)
        elif choice == '4e':
            function4(use_efficiency=True)
        elif choice == '5s':
            function5(show_database_summary=True)

'''The RPDUAA program (version 1.0) is provided under the 3-Clause BSD License.
Copyright <2021> <Haoran Zhang> <Peking University>
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met: 1. Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. 2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or other materials provided with the
distribution. 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.'''

'''Secondary structure analyses in the RPDUAA program require an auxillary executable of the DSSP program.
The auxillary DSSP program (https://swift.cmbi.umcn.nl/gv/dssp/) is provided under the Boost license below:
[Boost Software License - Version 1.0 - August 17th, 2003] Permission is hereby granted, free of charge, to any 
person or organization obtaining a copy of the software and accompanying documentation covered by this license 
(the "Software") to use, reproduce, display, distribute, execute, and transmit the Software, and to prepare 
derivative works of the Software, and to permit third-parties to whom the Software is furnished to do so, all 
subject to the following: The copyright notices in the Software and this entire statement, including the above 
license grant, this restriction and the following disclaimer, must be included in all copies of the Software, 
in whole or in part, and all derivative works of the Software, unless such copies or derivative works are solely 
in the form of machine-executable object code generated by a source language processor.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE 
COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN 
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.'''

'''Hidden function 1: When exploring Main Menu [1], add a single "_" before the pdb_name to download its 3 files.'''
'''Hidden function 2: Parallel to Main Menu [4], use "4a" to use absolute values of differences in preprocessings.'''
'''Hidden function 3: Parallel to Main Menu [4], use "4i" just to test the model with prospective laboratory data.'''
'''Hidden function 4: Parallel to Main Menu [4], use "4e" to predict the UAA efficiency along with the probability.'''
'''Hidden function 5: Parallel to Main Menu [5], use "5s" to show the summary of the database of known UAA sites.'''
'''Hidden function 6: Curves of 100 holdout validations. Input "1x/4x" in Main Menu [4] (export to hovdx100.csv).'''
'''Hidden function 7: Predict the whole database for review. Input "p1e" in Main Menu [4] (export to baseprob.csv).'''
'''Hidden function 8: Heatmap when using strategy 3 and predicting all for chains (heavy work, need a few seconds).'''
