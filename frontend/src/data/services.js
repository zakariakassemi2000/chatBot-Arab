/**
 * SHIFA AI — Services Configuration
 * Données centralisées pour toutes les cartes de service.
 * Chaque module est configurable : titre, icône, description, lien, état actif.
 * Links now use internal routes (React Router) instead of external URLs.
 */

export const quickServices = [
  {
    id: 'diagnostic',
    title: 'الذكاء التشخيصي',
    icon: '🧠',
    description: 'تحليل ذكي للأعراض وتوليد تقرير تشخيصي مبدئي',
    link: '/checkup',
    isActive: true,
    gradient: 'from-cyan-500/20 to-teal-500/10',
  },
  {
    id: 'assistant',
    title: 'المساعد الطبي',
    icon: '💬',
    description: 'محادثة ذكية لتقييم حالتك الصحية والإجابة عن أسئلتك',
    link: '/assistant',
    isActive: true,
    gradient: 'from-blue-500/20 to-indigo-500/10',
  },
  {
    id: 'checkup',
    title: 'فحص مبدئي',
    icon: '🩺',
    description: 'نظام تقييم سريري يعتمد على بياناتك وبيانات سريرية معتمدة',
    link: '/checkup',
    isActive: true,
    gradient: 'from-emerald-500/20 to-green-500/10',
  },
];

export const advancedModules = [
  {
    id: 'vision',
    title: 'مختبر الصور',
    icon: '🔬',
    description: 'تحليل صور الأشعة والرنين المغناطيسي',
    link: 'http://localhost:8501',
    isActive: true,
  },
  {
    id: 'ocr',
    title: 'ماسح الوصفات',
    icon: '📋',
    description: 'استخراج بيانات الوصفة الطبية بسرعة',
    link: 'http://localhost:8501',
    isActive: true,
  },
  {
    id: 'calculators',
    title: 'حاسبات سريرية',
    icon: '🧮',
    description: 'معادلات طبية دقيقة ومعتمدة دولياً',
    link: 'http://localhost:8501',
    isActive: true,
  },
  {
    id: 'mental',
    title: 'الصحة النفسية',
    icon: '🧘',
    description: 'دعم نفسي ومعرفي مخصص بالعربية',
    link: '/mental',
    isActive: true,
  },
  {
    id: 'voice',
    title: 'المساعد الصوتي',
    icon: '🎙️',
    description: 'تحدث مباشرة واحصل على إجابة فورية',
    link: 'http://localhost:8501',
    isActive: true,
  },
  {
    id: 'history',
    title: 'الأرشيف والسجلات',
    icon: '📜',
    description: 'مراجعة جلسات الاستشارة السابقة',
    link: 'http://localhost:8501',
    isActive: true,
  },
];

export const emergencyNumbers = {
  ambulance: { label: 'الإسعاف', number: '15', icon: '🚑' },
  police: { label: 'الشرطة', number: '19', icon: '🚓' },
};
