import cv2 as cv
import numpy as np

def gaussovo_jedro(d,h):
    #Enacba za izracun jedra po formuli 
    return np.exp(-d**2 / (2* h**2))

def kmeans(slika, k=3, iteracije=10): #Koncana K-means funkcija
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    visina, sirina, _ = slika.shape #Dobimo dimenzije slike
    slika = slika.astype(np.float32)
    
    # Pridobi vhodne parametre
    izbira = input("Vnesi način izbire centrov ('rocno' ali 'avto'): ").strip()
    dimenzija = int(input("Vnesi dimenzijo (3 za barve, 5 za barve+lokacija): "))
    T = int(input("Vnesi minimalno razdaljo T (npr. 30): "))
    
    # Priprava prostora znacilnic
    if dimenzija == 3:
        piksli = slika.reshape(-1, 3)
    elif dimenzija == 5:
        y, x = np.mgrid[0:visina, 0:sirina]
        # Normaliziraj koordinate
        lokacije = np.stack((x.flatten()/sirina, y.flatten()/visina), axis=1)
        barve = slika.reshape(-1, 3)
        piksli = np.hstack((barve, lokacije)).astype(np.float32)
    
    # Izračun začetnih centrov
    centri = izracunaj_centre(slika, izbira, dimenzija, T)
    
    # Glavna zanka K-means
    for _ in range(iteracije):
        # Izračunaj razdalje med piksli in centri (Evklidska razdalja)
        razdalje = np.linalg.norm(piksli[:, None] - centri, axis=2)
        # Dodeli piksle najbližjim centrom
        oznake = np.argmin(razdalje, axis=1)
        
        # Posodobi centre
        novi_centri = []
        for i in range(len(centri)):
            cluster = piksli[oznake == i]
            if len(cluster) > 0:
                novi_centri.append(cluster.mean(axis=0))
        centri = np.array(novi_centri)
    
    # Ustvari segmentirano sliko
    segmentirana = centri[oznake][:, :3].reshape(visina, sirina, 3).astype(np.uint8)
    return segmentirana


def meanshift(slika, h, dimenzija):
    """Izvede segmentacijo slike z uporabo mean-shift algoritma."""
    # Zmanjšaj sliko na 100x100 za hitrejšo obdelavo
    slika_mala = cv.resize(slika, (300, 300), interpolation=cv.INTER_AREA)
    visina, sirina, _ = slika_mala.shape
    slika_mala = slika_mala.astype(np.float32) / 255.0  # Normalizacija barv

    # Priprava prostora znacilnic
    if dimenzija == 3:
        #Vsak piksel predstavlja barvo RGB
        znacilnice = slika_mala.reshape(-1, 3)
    elif dimenzija == 5:
        #Ustvari dve matriki kjer y vsebuje vrsticni inedeks in x stolpicni za vsak piksel
        y, x = np.mgrid[0:visina, 0:sirina]
        #Normalizira koordinate zdruzi v tabelo kjer ima vsak x in y lokacijo
        lokacije = np.stack((x.flatten()/sirina, y.flatten()/visina), axis=1)
        barve = slika_mala.reshape(-1, 3)
        #Zdruzi barvno tabelo in lokacije v eno tabelo s petimi stolpci
        znacilnice = np.hstack((barve, lokacije))
    else:
        raise ValueError("Dimenzija mora biti 3 ali 5")

    # Parametri
    max_iteracije = 20
    epsilon = 0.1  # Toleranca za konvergenco
    min_cd = h / 2  # Parameter za združevanje centrov
    
    # Nakljuceno izbere 15% vseh pikslov kot zacetne tocke da se pohitri
    st_vzorcev = int(len(znacilnice) * 0.15)
    indeksi = np.random.choice(len(znacilnice), st_vzorcev, replace=False)
    tocke_start = znacilnice[indeksi]
    
    # Prilagodimo h za barve
    h_barve = 0.1 if h > 1 else h
    
    # Glavna Mean-Shift zanka
    konvergirane_tocke = []
    #Za vsako zacetno tocko izvede iteracije da se najde koncni center
    for i, tocka in enumerate(tocke_start):
        iteracija = 0
        nova_tocka = tocka.copy()
        #prepreci da se zanka izvaja neskonco
        while iteracija < max_iteracije:
            # Izračun za 5D prostor
            if dimenzija == 5:
                #Evklidska razdalje med barvami trenutne tocke in vsemi piksli
                razdalje_barve = np.sum((znacilnice[:, :3] - nova_tocka[:3])**2, axis=1)
                #Evklidska razdalja med normaliziranimi lokacijami
                razdalje_prostor = np.sum((znacilnice[:, 3:] - nova_tocka[3:])**2, axis=1)
                #Kombinacija utezi za bravo in prostor z gaussovim jedrom
                utezi = gaussovo_jedro(razdalje_barve, h_barve) * gaussovo_jedro(razdalje_prostor, h)
            else: #Izracub 3D
                #Evklidska razdalje med barvami trenutne tocke in vsemi piksli
                razdalje = np.sum((znacilnice - nova_tocka)**2, axis=1)
                utezi = gaussovo_jedro(razdalje, h_barve)
            
            # Izračun premika
            vsota_utezi = np.sum(utezi)
            #Vstavi iteracijo ce ni vplvnih pikslov
            if vsota_utezi == 0:
                break
                
            prejsnja_tocka = nova_tocka.copy()
            nova_tocka = np.sum(znacilnice * utezi[:, np.newaxis], axis=0) / vsota_utezi
            
            # Preveri konvergenco
            premik = np.linalg.norm(nova_tocka - prejsnja_tocka)
            #Toleranca za konvergenvo ce je premik manjsi tocka ne potrebuje vec iteracij
            if premik < epsilon:
                break
                
            iteracija += 1
            
        #Shrani koncne pozicije tock po konvergenci    
        konvergirane_tocke.append(nova_tocka)
    
    # Združevanje centrov
    centri = [] #Sezhnam koncnih cetrov
    #Za vsako konvergirano tocko iz iteracij
    for tocka in konvergirane_tocke:
        dodan = False
        #Preveri vse trenutno shranjene centre
        for i, center in enumerate(centri):
            if dimenzija == 5:
                #Izracuna barvno razdalo med trenutno tocko in centrom samo RGB
                razdalja_barve = np.linalg.norm(tocka[:3] - center[:3])
                #Ce je barva razdalja manjsa od minilalne kriticne razdalje
                if razdalja_barve < min_cd:
                    #Zdruzi center s tocko (Povprecje obeh)
                    centri[i] = (center + tocka) / 2
                    #Oznaci da smo tocko zdruzili
                    dodan = True
                    break
            else: #Ce delamo v 3D prostoru samo barve
                razdalja = np.linalg.norm(tocka - center)
                #Ce je razdalja manjsa od minimalne kriticne razdalje
                if razdalja < min_cd:
                    #Zdruzi center s tocko (Povprecje obeh)
                    centri[i] = (center + tocka) / 2
                    #Oznaci da smo tocko zdruzili
                    dodan = True
                    break
        #Ce tocke nismo zdruzili z nobenim centrom
        if not dodan:
            #Dodaj tocko kot nov center
            centri.append(tocka)
    
    # Zagotovi pravilno obliko
    centri = np.array(centri)
    #Ce ni bio najdenih dobenih centrov vrni originalno sliko
    if len(centri) == 0:
        return slika_mala
    # Ce je center 1D array ga pretvori v 2D (1, stevilo_lastnosti)
    if centri.ndim == 1:
        centri = centri.reshape(1, -1)
    
    # Dodelitev pikslov centrom in racunanje evklidske razdalje med piksli in centri
    razdalje = np.sqrt(np.sum((znacilnice[:, np.newaxis] - centri)**2, axis=2))
    #Dodeli vsakemu piksli oznako najbljizjega centra
    oznake = np.argmin(razdalje, axis=1)
    
    # Ustvari segmentirano sliko
    if dimenzija == 3:
        #Uporabi vse lastnosti centrov
        barve_centrov = centri
    else:
        #uporabi samo 3 prve lastnosti centrov BGR
        barve_centrov = centri[:, :3]
    #Preslikaj oznake v barve centrov in prebolikuje originalno obliko slike
    segmentirana = barve_centrov[oznake].reshape(visina, sirina, 3)
    #Pretvori normalizirane barve (0-1) nazaj v 255 in celastevila
    segmentirana = (segmentirana * 255).astype(np.uint8)

    #Vrne segmentirano sliko
    return segmentirana



def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    visina, sirina, _ = slika.shape #Dobimo dimenzije slike

    #Priprava prostora znacilnic
    if dimenzija_centra == 3:
        #3D uporabimo samo BGR
        znacilnice = slika.reshape(-1, 3).astype(np.float32)
    elif dimenzija_centra == 5:
        #5D uporabimo BGR in se x pa y
        y, x = np.mgrid[0:visina, 0:sirina] #Ustvari mrezo koordinat
        #Normalizcija koordinat
        lokacije = np.stack((x.flatten()/sirina, y.flatten()/visina), axis=1)
        barve = slika.reshape(-1, 3)
        znacilnice = np.hstack((barve, lokacije)).astype(np.float32)
        #Zdruzi barve in lokacije
    else:
        raise ValueError("dimenzija_centra mora biti 3 ali 5")

    centri = []
    st_centrov = 3  # Vedno izberemo 3 centre, ker uporabljamo samo 3 centre pri kmeans funkciji
    
    if izbira == "rocno":
        # Zagotovimo, da uporabljamo BGR za prikaz (OpenCV privzeto)
        slika_prikaz = slika.copy().astype(np.uint8)
        print(f"Kliknite na sliko, da izberete {st_centrov} centrov (ESC za konec).")

        def klik(event, x, y, flags, param):
            #Funkcija ki se sprozi ob kliku z misko
            if event == cv.EVENT_LBUTTONDOWN and len(centri) < st_centrov:
                if dimenzija_centra == 3:
                    #Za 3D vzame barvo izbranega piksla
                    center = slika[y, x].astype(np.float32)
                else:
                    #Za 5D vzame barvo in lokacijo piksla
                    barva = slika[y, x].astype(np.float32)
                    lokacija = np.array([x/sirina, y/visina], dtype=np.float32)
                    center = np.concatenate((barva, lokacija))
                centri.append(center) #Center v seznamu
                print(f"Izbran center {len(centri)}: {center}")
                cv.circle(slika_prikaz, (x, y), 5, (0, 255, 0), -1)
                cv.imshow("Izberi centre", slika_prikaz)
        #Nastavi okno in callback za klike
        cv.namedWindow("Izberi centre")
        cv.setMouseCallback("Izberi centre", klik)
        #Prikazuje sliko dokler niso izbrani vsi centri ali stisnemo ESC
        while len(centri) < st_centrov:
            cv.imshow("Izberi centre", slika_prikaz)
            if cv.waitKey(1) == 27:  # ESC
                break
        cv.destroyAllWindows()

    #Avtomatska izbira z upoštevanjem praga T
    elif izbira == "avto":
        st_vzorcev = znacilnice.shape[0] #Stevilo vseh kandidatov
        rng = np.random.default_rng() #Generator nakljucnih stevil
        poskusi = 0
        max_poskusov = 1000 #Prepreci neskoncno zanko

        while len(centri) < st_centrov and poskusi < max_poskusov:
            #Nakljucno izberi center
            kandidat = znacilnice[rng.integers(0, st_vzorcev)]
            #Preveri ali je dovolj oddaljen od ze izbranih centrov
            if all(np.linalg.norm(kandidat - c) > T for c in centri):
                centri.append(kandidat)
            poskusi += 1

        if len(centri) < st_centrov:
            raise RuntimeError(f"Ne najdem {st_centrov} centrov z T={T}!")

    else:
        raise ValueError("Neveljavna izbira. Dovoljeno: 'rocno' ali 'avto'")

    #Vrni center kot numpy array
    return np.array(centri, dtype=np.float32)



if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    slika = cv.imread("./.utils/zelenjava.jpg")
    if slika is None:
        print("Napaka: Slika ni bila najdena!")
        exit()

    # Izbira algoritma
    algoritem = input("Izberi algoritem (k za K-means, m za Mean-Shift): ").strip().lower()

    if algoritem == 'k':
        segmentirana = kmeans(slika, k=3, iteracije=10)
        naslov = "K-means Segmentacija"
        # Prikaz originala in rezultata v polni velikosti
        slika_prikaz = cv.resize(slika, (300, 300))
        segmentirana_prikaz = cv.resize(segmentirana, (300, 300))
    elif algoritem == 'm':
        h = float(input("Vnesi parameter širine okna (h): "))
        dimenzija = int(input("Vnesi dimenzijo (3 za barve, 5 za barve+lokacija): "))
        print("Izvajam Mean-Shift segmentacijo...")
        segmentirana = meanshift(slika, h, dimenzija)
        naslov = "Mean-Shift Segmentacija"
        # Prikaz originala 100x100 in rezultata 300x300
        slika_prikaz = cv.resize(slika, (300, 300))
        segmentirana_prikaz = cv.resize(segmentirana, (300, 300), interpolation=cv.INTER_NEAREST)
    else:
        print("Napačna izbira algoritma!")
        exit()

    cv.imshow("Original", slika_prikaz)
    cv.imshow(naslov, segmentirana_prikaz)
    cv.waitKey(0)
    cv.destroyAllWindows()